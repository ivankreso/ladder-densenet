import os, re
import pickle
import tensorflow as tf
import numpy as np
#import cv2
from os.path import join

import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope
#import skimage as ski
#import skimage.io

import libs.cylib as cylib
import train_helper
import losses
import eval_helper
import datasets.voc2012.reader as reader
from datasets.voc2012.dataset import Dataset


FLAGS = tf.app.flags.FLAGS
subset_dir = '/home/kivan/datasets/VOC2012/ImageSets/Segmentation/'
#subset_dir = '/home/kivan/datasets/voc2012_aug/'
dataset_dir = '/home/kivan/datasets/voc2012_aug/tensorflow/'
#tf.app.flags.DEFINE_string('dataset_dir', DATASET_DIR, '')
print('Dataset dir: ' + dataset_dir)

# RGB
data_mean = [116.49585869, 112.43425923, 103.19996733]
data_std = [60.37073962, 59.39268441, 60.74823033]

if FLAGS.no_valid:
  train_dataset = Dataset(dataset_dir, join(subset_dir, 'trainval.txt'), 'trainval')
else:
  train_dataset = Dataset(dataset_dir, join(subset_dir, 'train.txt'), 'train')
  valid_dataset = Dataset(dataset_dir, join(subset_dir, 'val.txt'), 'val')


print('Num training examples = ', train_dataset.num_examples())

#model_depth = 121
#block_sizes = [6,12,24,16]
model_depth = 169
block_sizes = [6,12,32,32]

imagenet_init = True
#imagenet_init = False
init_dir = '/home/kivan/datasets/pretrained/dense_net/'
apply_jitter = True
#apply_jitter = False
jitter_scale = False
#jitter_scale = True
pool_func = layers.avg_pool2d
#pool_func = layers.max_pool2d
known_shape = True

train_step_iter = 0

weight_decay = 1e-4
#weight_decay = 4e-5
#weight_decay = 2e-4
#init_func = layers.variance_scaling_initializer(mode='FAN_OUT')
init_func = layers.variance_scaling_initializer()

context_size = 512
growth = 32
compression = 0.5
growth_up = 32

use_dropout = False
#use_dropout = True
keep_prob = 0.8

# must be false if BN is frozen
fused_batch_norm = True
#fused_batch_norm = False

#data_format = 'NCHW'
#maps_dim = 1
#height_dim = 2

data_format = 'NHWC'
maps_dim = 3
height_dim = 1


bn_params = {
  # Decay for the moving averages.
  'decay': 0.9,
  'center': True,
  'scale': True,
  # epsilon to prevent 0s in variance.
  'epsilon': 1e-5,
  # None to force the updates
  'updates_collections': None,
  'fused': fused_batch_norm,
  'data_format': data_format,
  'is_training': True
}


def evaluate(name, sess, epoch_num, run_ops, data):
  loss_val, accuracy, iou, recall, precision = eval_helper.evaluate_segmentation_voc2012(
      sess, epoch_num, run_ops, valid_dataset)
  is_best = False
  if iou > data['best_iou'][0]:
    is_best = True
    data['best_iou'] = [iou, epoch_num]
  data['iou'] += [iou]
  data['acc'] += [accuracy]
  data['loss'] += [loss_val]
  return is_best


def start_epoch(train_data):
  global train_loss_arr, train_conf_mat
  train_conf_mat = np.ascontiguousarray(
      np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64))
  train_loss_arr = []
  train_data['lr'].append(lr.eval())


def end_epoch(train_data):
  pixacc, iou, _, _, _ = eval_helper.compute_errors(
      train_conf_mat, 'Train', train_dataset.class_info)
  is_best = False
  if len(train_data['iou']) > 0 and iou > max(train_data['iou']):
    is_best = True
  train_data['iou'].append(iou)
  train_data['acc'].append(pixacc)
  train_loss_val = np.mean(train_loss_arr)
  train_data['loss'].append(train_loss_val)
  return is_best


def update_stats(ret_val):
  global train_loss_arr
  loss_val = ret_val[0]
  yp = ret_val[1]
  yt = ret_val[2]
  train_loss_arr.append(loss_val)
  yp = yp.argmax(3).astype(np.int32)
  cylib.collect_confusion_matrix(yp.reshape(-1), yt.reshape(-1), train_conf_mat)


def plot_results(train_data, valid_data):
  eval_helper.plot_training_progress(os.path.join(FLAGS.train_dir, 'stats'),
                                     train_data, valid_data)


def print_results(train_data, valid_data):
  print('\nBest train IOU = %.2f' % max(train_data['iou']))
  print('Best validation IOU = %.2f (epoch %d)\n' % tuple(valid_data['best_iou']))


def init_eval_data():
  train_data = {}
  valid_data = {}
  train_data['lr'] = []
  train_data['loss'] = []
  train_data['iou'] = []
  train_data['acc'] = []
  train_data['best_iou'] = [0, 0]
  valid_data['best_iou'] = [0, 0]
  valid_data['loss'] = []
  valid_data['iou'] = []
  valid_data['acc'] = []
  return train_data, valid_data


def normalize_input(img):
  with tf.name_scope('input'), tf.device('/gpu:0'):
    if data_format == 'NCHW':
      img = tf.transpose(img, perm=[0,3,1,2])
      mean = tf.constant(data_mean, dtype=tf.float32, shape=[1,3,1,1])
      std = tf.constant(data_std, dtype=tf.float32, shape=[1,3,1,1])
    else:
      mean = data_mean
      std = data_std
    img = (img - mean) / std
    return img


def resize_tensor(net, shape, name):
  if data_format == 'NCHW':
    net = tf.transpose(net, perm=[0,2,3,1])
  net = tf.image.resize_bilinear(net, shape, name=name)
  if data_format == 'NCHW':
    net = tf.transpose(net, perm=[0,3,1,2])
  return net


def refine(net, skip_data, is_training):
  skip_net = skip_data[0]
  num_layers = skip_data[1]
  growth = skip_data[2]
  block_name = skip_data[3]

  #size_top = top_layer.get_shape()[maps_dim].value
  #skip_width = skip_layer.get_shape()[2].value
  #if top_height != skip_height or top_width != skip_width:
    #print(top_height, skip_height)
    #assert(2*top_height == skip_height)
  
  #TODO try convolution2d_transpose
  #up_shape = tf.shape(skip_net)[height_dim:height_dim+2]
  with tf.variable_scope(block_name):
    if known_shape:
      up_shape = skip_net.get_shape().as_list()[height_dim:height_dim+2]
    else:
      up_shape = tf.shape(skip_net)[height_dim:height_dim+2]
    shape_info = net.get_shape().as_list()
    print(net)
    net = resize_tensor(net, up_shape, name='upsample')
    print(net)
    if not known_shape:
      print(shape_info)
      shape_info[height_dim] = None
      shape_info[height_dim+1] = None
      net.set_shape(shape_info)
    print('\nup = ', net)
    print('skip = ', skip_net)
    #print(skip_data)
    return upsample(net, skip_net, num_layers, growth, is_training, 'dense_block')


def BNReluConv(net, num_filters, name, k=3, rate=1, first=False, concat=None):
  with arg_scope([layers.conv2d],
      data_format=data_format, stride=1, padding='SAME', activation_fn=None,
      normalizer_fn=None, normalizer_params=None,
      weights_initializer=init_func, biases_initializer=None,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.variable_scope(name):
      # TODO check this
      relu = None
      if not first:
        # TODO try Relu -> BN
        net = tf.contrib.layers.batch_norm(net, **bn_params)
        net = tf.nn.relu(net)
        relu = net
      if concat is not None:
        net = tf.concat([net, concat], maps_dim)
        print('c ', net)
      net = layers.conv2d(net, num_filters, kernel_size=k, rate=rate)
    return net


def _pyramid_pooling(net, size, num_pools=3):
  print('Pyramid context pooling')
  with tf.variable_scope('pyramid_context_pooling'):
    if known_shape:
      shape = net.get_shape().as_list()
    else:
      shape = tf.shape(net)
    print('shape = ', shape)
    up_size = shape[height_dim:height_dim+2]
    shape_info = net.get_shape().as_list()
    num_maps = net.get_shape().as_list()[maps_dim]
    #grid_size = [6, 3, 2, 1]
    pool_dim = int(round(num_maps / num_pools))
    concat_lst = [net]
    for i in range(num_pools):
      #pool = layers.avg_pool2d(net, kernel_size=[kh, kw], stride=[kh, kw], padding='SAME')
      #pool = layers.avg_pool2d(net, kernel_size=[kh, kh], stride=[kh, kh], padding='SAME')
      print('before pool = ', net)
      net = layers.avg_pool2d(net, 2, 2, padding='SAME', data_format=data_format)
      print(net)
      pool = BNReluConv(net, pool_dim, k=1, name='bottleneck'+str(i))
      #pool = tf.image.resize_bilinear(pool, [height, width], name='resize_score')
      pool = resize_tensor(pool, up_size, name='upsample_level_'+str(i))
      concat_lst.append(pool)
    net = tf.concat(concat_lst, maps_dim)
    print('Pyramid pooling out: ', net)
    #net = BNReluConv(net, 512, k=3, name='bottleneck_out')
    net = BNReluConv(net, size, k=3, name='bottleneck_out')
    return net


def layer(net, num_filters, name, is_training, first):
  with tf.variable_scope(name):
    net = BNReluConv(net, 4*num_filters, 'bottleneck', k=1, first=first)
    net = BNReluConv(net, num_filters, 'conv', k=3)
    if use_dropout and is_training: 
      net = tf.nn.dropout(net, keep_prob=keep_prob)
  return net


def dense_block(net, size, growth, name, is_training=False, first=False,
                split=False, rate=1):
  with tf.variable_scope(name):
    for i in range(size):
      x = net
      #net, first_relu = layer(net, k, 'layer'+str(i), is_training, first=first)
      net = layer(net, growth, 'layer'+str(i), is_training, first=first)
      net = tf.concat([x, net], maps_dim)
      if first:
        first = False
      if split and i == (size // 2) - 1:
        split_out = net
        print('Split shape = ', net)
        if rate == 1:
          net = pool_func(net, 2, stride=2, padding='SAME', data_format=data_format)
        else:
          paddings, crops = tf.required_space_to_batch_paddings(image_size(net),
              [rate,rate])
          net = tf.space_to_batch(net, paddings=paddings, block_size=rate)
  if split and rate > 1:
    net = tf.batch_to_space(net, crops=crops, block_size=rate)
  print('Dense block out: ', net)
  if split:
    return net, split_out
  return net

def dense_block_multigpu(net, size, growth, name, is_training=False, first=False, split=False):
  with tf.variable_scope(name):
    for i in range(size):
      #if i < size//2:
      #if i < 6:
      #if i < 3:

      if i < 12:
        gpu = '/gpu:0'
      else:
        gpu = '/gpu:1'
      with tf.device(gpu):
        x = net
        #net, first_relu = layer(net, k, 'layer'+str(i), is_training, first=first)
        net = layer(net, growth, 'layer'+str(i), is_training, first=first)
        net = tf.concat([x, net], maps_dim)
        if first:
          first = False
        if split and i == (size // 2) - 1:
          split_out = net
          print('Split shape = ', net)
          net = pool_func(net, 2, stride=2, padding='SAME', data_format=data_format)
  print('Dense block out: ', net)
  if split == True:
    return net, split_out
  return net

#growth_up = 32
#up_sizes = [2,2,4,4]
#def dense_block_upsample(net, skip_net, size, growth, name):
#  with tf.variable_scope(name):
#    net = tf.concat([net, skip_net], maps_dim)
#    num_filters = net.get_shape().as_list()[maps_dim]
#    #num_filters = int(round(num_filters*compression))
#    num_filters = int(round(num_filters*compression))
#    #num_filters = int(round(num_filters*0.3))
#    # TODO try 3 vs 1
#    net = BNReluConv(net, num_filters, 'bottleneck', k=1)
#    #net = BNReluConv(net, num_filters, name+'_bottleneck', k=3)
#    print('after bottleneck = ', net)
#    for i in range(size):
#      x = net
#      net = BNReluConv(net, growth, 'layer'+str(i))
#      net = tf.concat([x, net], maps_dim)
#  return net
#  #return dense_block(net, size, growth, name)


# old refine
##up_sizes = [128,128,512,512]
#up_sizes = [256,256,512,512]
#up_sizes = [196,256,384,512]
#up_sizes = [128,128,256,512] # 2gpus
#up_sizes = [64,128,256,512] # 2gpus
#up_sizes = [128,128,256,384,512]
#up_sizes = [64,128,256,384,512]
#up_sizes = [256,256,384,512]
#up_sizes = [128,128,256,256]

# best
#up_sizes = [64,128,128,256]


def dense_block_upsample_worse(net, skip_net, size, growth, name):
  with tf.variable_scope(name):
    net = tf.concat([net, skip_net], maps_dim)
    #new_size = net.get_shape().as_list()[height_dim:height_dim+2]
    #depth = resize_tensor(depth, new_size, 'resize_depth')
    #net = tf.concat([net, skip_net, depth], maps_dim)

    num_filters = net.get_shape().as_list()[maps_dim]
    print(net)
    num_filters = int(round(num_filters*compression))
    #num_filters = int(round(num_filters*compression/2))
    #num_filters = int(round(num_filters*0.3))
    net = BNReluConv(net, num_filters, 'bottleneck', k=1)
    #net = BNReluConv(net, num_filters, 'bottleneck', k=3)
    #net = tf.concat([net, depth], maps_dim)
    #net = BNReluConv(net, num_filters, 'bottleneck', k=3)
    print('after bottleneck = ', net)
    net = BNReluConv(net, size, 'layer')
  return net
  #return dense_block(net, size, growth, name)

#up_sizes = [256,256,256,384]

# try stronger upsampling
#up_sizes = [64,128,256,512] # good
#up_sizes = [128,256,256,256]

#up_sizes = [128,256,256,512] # good
#up_sizes = [128,128,256,256]
#up_sizes = [128,196,256,384]
#up_sizes = [128,196,256,384,512]
#up_sizes = [64,128,128,128,256]
#up_sizes = [64,64,64,64,64]

#up_sizes = [256,256,512,512] # good
#up_sizes = [128,256,384,512] # 0.5% worse then up
#up_sizes = [32,64,128,256]

up_sizes = [128,128,128,128,128]
#up_sizes = [128,128,256,256,256]

def upsample(net, skip_net, size, growth, is_training, name):
  with tf.variable_scope(name):
    # TODO
    num_filters = net.get_shape().as_list()[maps_dim]
    skip_net = BNReluConv(skip_net, num_filters, 'bottleneck', k=1)
    net = tf.concat([net, skip_net], maps_dim)
    #net = net + skip_net
    #net = BNReluConv(net, num_filters, 'bottleneck', k=3)
    print('after concat = ', net)
    net = BNReluConv(net, size, 'layer')
  return net

# works the same as simple
def upsample_dense(net, skip_net, size, growth, is_training, name):
  with tf.variable_scope(name):
    num_filters = net.get_shape().as_list()[maps_dim]
    skip_net = BNReluConv(skip_net, num_filters, 'skip_bottleneck', k=1)
    net = tf.concat([net, skip_net], maps_dim)
    net = dense_block(net, 4, growth, 'dense_block', is_training)
    #net = BNReluConv(net, num_filters, 'bottleneck', k=3)
    print('after dense block = ', net)
    net = BNReluConv(net, size, 'bottleneck', k=1)
  return net

def transition(net, compression, name, stride=2, pool=True):
  with tf.variable_scope(name):
    net = tf.contrib.layers.batch_norm(net, **bn_params)
    net = tf.nn.relu(net)
    num_filters = net.get_shape().as_list()[maps_dim]
    num_filters = int(round(num_filters*compression))
    net = layers.conv2d(net, num_filters, kernel_size=1)
    skip_layer = net
    # avg works little better on small res
    if pool:
      net = pool_func(net, 2, stride=stride, data_format=data_format, padding='SAME')
  print('Transition: ', net)
  return net, skip_layer


def dense_block_context(net):
  print('Dense context')
  with tf.variable_scope('block_context'):
    outputs = []
    size = 8
    #size = 4
    #size = 6
    for i in range(size):
      x = net
      net = BNReluConv(net, 64, 'layer'+str(i))
      #net = BNReluConv(net, 128, 'layer'+str(i))
      outputs.append(net)
      if i < size - 1:
        net = tf.concat([x, net], maps_dim)
    net = tf.concat(outputs, maps_dim)
  return net


def _build(image, is_training=False):
  #image = tf.Print(image, [tf.shape(image)], message='img_shape = ', summarize=10)
  bn_params['is_training'] = is_training
  with arg_scope([layers.conv2d],
      data_format=data_format, stride=1, padding='SAME', activation_fn=None,
      normalizer_fn=None, normalizer_params=None,
      weights_initializer=init_func, biases_initializer=None,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.variable_scope('conv0'):
      net = layers.conv2d(image, 2*growth, 7, stride=2)
      #net = layers.conv2d(image, 2*growth, 7, stride=1)
      # TODO
      net = tf.contrib.layers.batch_norm(net, **bn_params)
      net = tf.nn.relu(net)

    net = layers.max_pool2d(net, 2, stride=2, padding='SAME',
                            data_format=data_format, scope='pool0')

    skip_layers = []

    # no diff with double BN from orig densenet, first=True
    net = dense_block(net, block_sizes[0], growth, 'block0', is_training, first=True)
    #net, skip = dense_block(net, block_sizes[0], growth, 'block0', is_training,
    #    first=True, split=True)
    #skip_layers.append([skip, 256, growth_up, 'block0_mid_refine', depth])
    #skip_layers.append([skip, up_sizes[0], growth_up, 'block0_mid_refine'])
    skip_layers.append([net, up_sizes[0], growth_up, 'block0_refine'])
    net, _ = transition(net, compression, 'block0/transition')
    #skip_layers.append([skip, up_sizes[0], growth_up, 'block0_refine'])

    #net, skip = dense_block(net, block_sizes[1], growth, 'block1', is_training, split=True)
    #skip_layers.append([skip, up_sizes[1], growth_up, 'block1_mid_refine'])
    net = dense_block(net, block_sizes[1], growth, 'block1', is_training)

    skip_layers.append([net, up_sizes[1], growth_up, 'block1_refine'])
    net, _ = transition(net, compression, 'block1/transition')

    #context_pool_num = 3
    #net, skip = dense_block(net, block_sizes[2], growth, 'block2', is_training, split=True)
    #skip_layers.append([skip, up_sizes[2], growth_up, 'block2_mid_refine'])
    #skip_layers.append([net, up_sizes[3], growth_up, 'block2_refine'])

    net = dense_block(net, block_sizes[2], growth, 'block2', is_training)
    skip_layers.append([net, up_sizes[2], growth_up, 'block2_refine'])
    net, _ = transition(net, compression, 'block2/transition')

    is_dilated = False
    #is_dilated = True
    #bsz = 2
    #net, _ = transition(net, compression, 'block1/transition', stride=1)
    #paddings, crops = tf.required_space_to_batch_paddings(image_size(net), [bsz, bsz])
    #net = tf.space_to_batch(net, paddings=paddings, block_size=bsz)
    #net = dense_block(net, block_sizes[2], growth, 'block2', is_training)
    #net, _ = transition(net, compression, 'block2/transition', stride=1)
    #net = tf.batch_to_space(net, crops=crops, block_size=bsz)

    #bsz = 2
    #bsz = 4
    #paddings, crops = tf.required_space_to_batch_paddings(image_size(net), [bsz, bsz])
    #net = tf.space_to_batch(net, paddings=paddings, block_size=bsz)
    #net = dense_block(net, block_sizes[3], growth, 'block3', is_training)
    #net = tf.batch_to_space(net, crops=crops, block_size=bsz)
    net, skip = dense_block(net, block_sizes[3], growth, 'block3', is_training, split=True)
    skip_layers.append([skip, up_sizes[-1], growth_up, 'block3_refine'])
    #skip = tf.batch_to_space(skip, crops=crops, block_size=bsz)

    #net, skip = dense_block(net, block_sizes[3], growth, 'block3', is_training,
    #                        split=True, rate=2)

    with tf.variable_scope('head'):
      print('out = ', net)
      #skip_layers.append([net, up_sizes[-1], growth_up, 'block3_refine'])
      #net, _ = transition(net, compression, 'block3/transition')
      #net = dense_block(net, block_sizes[3], growth, 'block4', is_training)
      net = BNReluConv(net, 512, 'bottleneck', k=1)
      ## 0.4 better with rate=2
      net = BNReluConv(net, 128, 'context', k=3, rate=2)
      ##net = BNReluConv(net, 128, 'context', k=3)

      #net = BNReluConv(net, 256, 'bottleneck', k=1)
      ##net = _pyramid_pooling(net, 256, num_pools=4)
      #net = _pyramid_pooling(net, 256, num_pools=3)
      ## SPP has dropout here
      #if is_training:
      #  net = tf.nn.dropout(net, keep_prob=0.9)

      #skip_layers.append([net, up_sizes[-1], growth_up, 'block3_refine'])
      #net, _ = transition(net, compression, 'block3/transition')
      #net, _ = transition(net, compression, 'block3/transition', stride)
      ####bsz = 4
      #bsz = 2
      #paddings, crops = tf.required_space_to_batch_paddings(image_size(net), [bsz, bsz])
      #net = tf.space_to_batch(net, paddings=paddings, block_size=bsz)
      #net = dense_block(net, block_sizes[3], growth, 'block4', is_training)
      ##print(net)
      ##net = BNReluConv(net, 256, 'bottleneck', k=1)
      ###net = BNReluConv(net, 128, 'bottleneck', k=1)
      #net = BNReluConv(net, 256, 'context', k=3, rate=2)
      #net = BNReluConv(net, 256, 'context', k=3, rate=2)
      #net = BNReluConv(net, 128, 'context', k=3)
      #net = tf.batch_to_space(net, crops=crops, block_size=bsz)

      #net = dense_block_context(net)
      #net = BNReluConv(net, context_size, 'context_conv', k=3)
      #context_pool_num = 4

      #print('dense context')
      #print('7x7')
      #net = BNReluConv(net, context_size, 'context_conv', k=7)
      #net = BNReluConv(net, context_size, 'context_conv', k=7, rate=2)
      #net = BNReluConv(net, context_size, 'context_conv', k=3, rate=2)
      #in_shape = net.get_shape().as_list()
      #in_shape[maps_dim] = context_size
      #net.set_shape(in_shape)
      #net = BNReluConv(net, context_size, 'context_conv', k=5)
      #final_h = net.get_shape().as_list()[height_dim]
      print('Before upsampling: ', net)

      all_logits = [net]
      for skip_layer in reversed(skip_layers):
        net = refine(net, skip_layer, is_training)
        all_logits.append(net)
        print('after upsampling = ', net)
      if not is_dilated:
        all_logits = [all_logits[0], all_logits[-1]]
      else:
        all_logits = [all_logits[-1]]

  with tf.variable_scope('head'):
    for i, logits in enumerate(all_logits):
      with tf.variable_scope('logits_'+str(i)):
      # FIX
      #net = tf.nn.relu(layers.batch_norm(net, **bn_params))
      #logits = layers.conv2d(net, FLAGS.num_classes, 1, activation_fn=None,
      #                       data_format=data_format)
        logits = layers.conv2d(tf.nn.relu(logits), FLAGS.num_classes, 1,
                               activation_fn=None, data_format=data_format)

        if data_format == 'NCHW':
          logits = tf.transpose(logits, perm=[0,2,3,1])
        input_shape = tf.shape(image)[height_dim:height_dim+2]
        logits = tf.image.resize_bilinear(logits, input_shape, name='resize_logits')
        all_logits[i] = logits
    logits = all_logits.pop()
    return logits, all_logits

    #with tf.variable_scope('logits'):
    #  #net = tf.nn.relu(layers.batch_norm(net, **bn_params))
    #  net = tf.nn.relu(net)
    #  logits = layers.conv2d(net, FLAGS.num_classes, 1, activation_fn=None,
    #                         data_format=data_format)

    #with tf.variable_scope('mid_logits'):
    #  #mid_logits = tf.nn.relu(layers.batch_norm(mid_logits, **bn_params))
    #  mid_logits = tf.nn.relu(mid_logits)
    #  mid_logits = layers.conv2d(mid_logits, FLAGS.num_classes, 1, activation_fn=None,
    #                             data_format=data_format)

    #if data_format == 'NCHW':
    #  logits = tf.transpose(logits, perm=[0,2,3,1])
    #  mid_logits = tf.transpose(mid_logits, perm=[0,2,3,1])
    #input_shape = tf.shape(image)[height_dim:height_dim+2]
    #logits = tf.image.resize_bilinear(logits, input_shape, name='resize_logits')
    #mid_logits = tf.image.resize_bilinear(mid_logits, input_shape, name='resize_mid_logits')
    ##if data_format == 'NCHW':
    ##  top_layer = tf.transpose(top_layer, perm=[0,3,1,2])
    #return logits, mid_logits


def _build2gpu(image, is_training=False):
  #image = tf.Print(image, [tf.shape(image)], message='img_shape = ', summarize=10)
  bn_params['is_training'] = is_training
  with arg_scope([layers.conv2d],
      data_format=data_format, stride=1, padding='SAME', activation_fn=None,
      normalizer_fn=None, normalizer_params=None,
      weights_initializer=init_func, biases_initializer=None,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    gpu1 = '/gpu:0'
    gpu2 = '/gpu:1'
    with tf.device(gpu1):
      with tf.variable_scope('conv0'):
        net = layers.conv2d(image, 2*growth, 7, stride=2)
        #net = layers.conv2d(image, 2*growth, 7, stride=1)
        # TODO
        net = tf.contrib.layers.batch_norm(net, **bn_params)
        net = tf.nn.relu(net)

      net = layers.max_pool2d(net, 2, stride=2, padding='SAME',
                              data_format=data_format, scope='pool0')

      skip_layers = []

      # no diff with double BN from orig densenet, first=True
      net = dense_block(net, block_sizes[0], growth, 'block0', is_training, first=True)
      #net, skip = dense_block(net, block_sizes[0], growth, 'block0', is_training,
      #    first=True, split=True)
      #skip_layers.append([skip, 256, growth_up, 'block0_mid_refine', depth])
      #skip_layers.append([skip, up_sizes[0], growth_up, 'block0_mid_refine'])
      skip_layers.append([net, up_sizes[0], growth_up, 'block0_refine'])
      net, _ = transition(net, compression, 'block0/transition')
      #skip_layers.append([skip, up_sizes[0], growth_up, 'block0_refine'])

      #net, skip = dense_block(net, block_sizes[1], growth, 'block1', is_training, split=True)
      #skip_layers.append([skip, up_sizes[1], growth_up, 'block1_mid_refine'])
    #with tf.device(gpu2):
      net = dense_block(net, block_sizes[1], growth, 'block1', is_training)
      skip_layers.append([net, up_sizes[1], growth_up, 'block1_refine'])
      net, _ = transition(net, compression, 'block1/transition')
      #skip_layers.append([skip, up_sizes[1], growth_up, 'block1_refine'])

      # works the same with split, not 100%
      #context_pool_num = 3
      net, skip = dense_block(net, block_sizes[2], growth, 'block2', is_training, split=True)
      skip_layers.append([skip, up_sizes[2], growth_up, 'block2_mid_refine'])
      context_pool_num = 4
    with tf.device(gpu2):
      #net = dense_block(net, block_sizes[2], growth, 'block2', is_training)
      skip_layers.append([net, up_sizes[3], growth_up, 'block2_refine'])
      #skip_layers.append([net, up_sizes[2], growth_up, 'block2_refine'])
      net, _ = transition(net, compression, 'block2/transition')

      #net = dense_block(net, block_sizes[3], growth, 'block3', is_training)
      net, skip = dense_block(net, block_sizes[3], growth, 'block3', is_training, split=True)
      context_pool_num = 3
      skip_layers.append([skip, up_sizes[-1], growth_up, 'block3_refine'])

      with tf.variable_scope('head'):
        #net = BNReluConv(net, 512, 'bottleneck', k=1)
        net = BNReluConv(net, 512, 'bottleneck', k=1)
        net = BNReluConv(net, 128, 'context', k=3, rate=2)
        #net = BNReluConv(net, 128, 'bottleneck', k=1)
        #net = dense_block_context(net)
        #net = _pyramid_pooling(net, size=context_pool_num)
        #net = BNReluConv(net, context_size, 'context_conv', k=3)
        # SPP has dropout here
        #if is_training:
        #  net = tf.nn.dropout(net, keep_prob=0.9)

        #print('dense context')
        #print('7x7')
        #net = BNReluConv(net, context_size, 'context_conv', k=7)
        #net = BNReluConv(net, context_size, 'context_conv', k=7, rate=2)
        #net = BNReluConv(net, context_size, 'context_conv', k=3, rate=2)
        #in_shape = net.get_shape().as_list()
        #in_shape[maps_dim] = context_size
        #net.set_shape(in_shape)
        #net = BNReluConv(net, context_size, 'context_conv', k=5)
        #final_h = net.get_shape().as_list()[height_dim]
        print('Before upsampling: ', net)

        all_logits = [net]
        for skip_layer in reversed(skip_layers):
          net = refine(net, skip_layer)
          all_logits.append(net)
          print('after upsampling = ', net)

        all_logits = [all_logits[0], all_logits[-1]]
        #all_logits = [all_logits[1], all_logits[-1]]
        #all_logits = [all_logits[2], all_logits[-1]]

  with tf.device(gpu2), tf.variable_scope('head'):
      for i, logits in enumerate(all_logits):
        with tf.variable_scope('logits_'+str(i)):
        # FIX
        #net = tf.nn.relu(layers.batch_norm(net, **bn_params))
        #logits = layers.conv2d(net, FLAGS.num_classes, 1, activation_fn=None,
        #                       data_format=data_format)
          logits = layers.conv2d(tf.nn.relu(logits), FLAGS.num_classes, 1,
                                 activation_fn=None, data_format=data_format)

          if data_format == 'NCHW':
            logits = tf.transpose(logits, perm=[0,2,3,1])
          input_shape = tf.shape(image)[height_dim:height_dim+2]
          logits = tf.image.resize_bilinear(logits, input_shape, name='resize_logits')
          all_logits[i] = logits
      logits = all_logits.pop()
      return logits, all_logits

def create_init_op(params):
  variables = tf.contrib.framework.get_variables()
  init_map = {}
  # clear head vars from imagenet
  remove_keys = []
  for key in params.keys():
    if 'head/' in key:
      print('delete ', key)
      remove_keys.append(key)
  for key in remove_keys:
    del params[key]

  for var in variables:
    name = var.name
    if name in params:
      #print(name, ' --> found init')
      #print(var)
      #print(params[name].shape)
      init_map[var.name] = params[name]
      del params[name]
    #else:
    #  print(name, ' --> init not found!')
  print('Unused: ', list(params.keys()))
  init_op, init_feed = tf.contrib.framework.assign_from_values(init_map)
  return init_op, init_feed


#def jitter(image, labels, weights):
def jitter(image, labels):
  with tf.name_scope('jitter'), tf.device('/cpu:0'):
    print('\nJittering enabled')
    global random_flip_tf, resize_width, resize_height
    #random_flip_tf = tf.placeholder(tf.bool, shape=(), name='random_flip')
    random_flip_tf = tf.placeholder(tf.bool, shape=(FLAGS.batch_size), name='random_flip')
    resize_width = tf.placeholder(tf.int32, shape=(), name='resize_width')
    resize_height = tf.placeholder(tf.int32, shape=(), name='resize_height')
    
    #image_split = tf.unstack(image, axis=0)
    #labels_split = tf.unstack(labels, axis=0)
    #weights_split = tf.unstack(weights, axis=0)
    out_img = []
    #out_weights = []
    out_labels = []
    for i in range(FLAGS.batch_size):
      out_img.append(tf.cond(random_flip_tf[i],
        lambda: tf.image.flip_left_right(image[i]),
        lambda: image[i]))
      out_labels.append(tf.cond(random_flip_tf[i],
        lambda: tf.image.flip_left_right(labels[i]),
        lambda: labels[i]))
      #out_weights.append(tf.cond(random_flip_tf[i],
      #  lambda: tf.image.flip_left_right(weights[i]),
      #  lambda: weights[i]))
    image = tf.stack(out_img, axis=0)
    labels = tf.stack(out_labels, axis=0)
    #weights = tf.stack(out_weights, axis=0)

    if jitter_scale:
      global known_shape
      known_shape = False
      image = tf.image.resize_bicubic(image, [resize_height, resize_width])
      #image = tf.image.resize_bilinear(image, [resize_height, resize_width])
      image = tf.round(image)
      image = tf.minimum(255.0, image)
      image = tf.maximum(0.0, image)
      labels = tf.image.resize_nearest_neighbor(labels, [resize_height, resize_width])
      # TODO is this safe for zero wgts?
      #weights = tf.image.resize_nearest_neighbor(weights, [resize_height, resize_width])
    #return image, labels, weights
    return image, labels


def _get_train_feed():
  global random_flip_tf, resize_width, resize_height
  #random_flip = int(np.random.choice(2, 1))
  random_flip = np.random.choice(2, FLAGS.batch_size).astype(np.bool)
  #resize_scale = np.random.uniform(0.5, 2)
  #resize_scale = np.random.uniform(0.4, 1.5)
  #resize_scale = np.random.uniform(0.5, 1.2)
  #min_resize = 0.7
  #max_resize = 1.3
  min_resize = 0.8
  max_resize = 1.2
  #min_resize = 0.9
  #max_resize = 1.1
  #max_resize = 1
  if train_step_iter == 0:
    resize_scale = max_resize
  else:
    resize_scale = np.random.uniform(min_resize, max_resize)
  width = np.int32(int(round(FLAGS.img_width * resize_scale)))
  height = np.int32(int(round(FLAGS.img_height * resize_scale)))
  feed_dict = {random_flip_tf:random_flip, resize_width:width, resize_height:height}
  return feed_dict


def build(mode):
  if mode == 'train':
    is_training = True
    reuse = False
    dataset = train_dataset
  elif mode == 'validation':
    is_training = False
    reuse = True
    dataset = valid_dataset

  with tf.variable_scope('', reuse=reuse):
    x, labels, num_labels, class_hist, img_names = \
      reader.inputs(dataset, is_training=is_training)
      #reader.inputs(dataset, is_training=is_training, num_epochs=FLAGS.max_epochs)

    if is_training and apply_jitter:
      x, labels = jitter(x, labels)
    image = x
    x = normalize_input(x)

    #logits = _build(x, depth, is_training)
    #total_loss = _loss(logits, labels, weights, is_training)
    #logits, mid_logits = _build(x, is_training)
    logits, aux_logits = _build(x, is_training)
    total_loss = _multiloss(logits, aux_logits, labels, class_hist, num_labels, is_training)

    if is_training and imagenet_init:
      init_path = init_dir + 'dense_net_' + str(model_depth) + '.pickle'
      with open(init_path, 'rb') as f:
        init_map = pickle.load(f)
      init_op, init_feed = create_init_op(init_map)
    else:
      init_op, init_feed = None, None
    train_run_ops = [total_loss, logits, labels, img_names]
    #train_run_ops = [total_loss, logits, labels, img_names, image]
    val_run_ops = [total_loss, logits, labels, img_names]
    if is_training:
      return train_run_ops, init_op, init_feed
    else:
      return val_run_ops


def inference(image, labels=None, constant_shape=True, is_training=False):
  global known_shape
  known_shape = constant_shape
  x = normalize_input(image)
  logits, aux_logits = _build(x, is_training=is_training)
  if labels:
    main_wgt = 0.7
    xent_loss = main_wgt * losses.weighted_cross_entropy_loss(logits, labels)
    xent_loss = (1-main_wgt) * losses.weighted_cross_entropy_loss(aux_logits, labels)
    return logits, aux_logits, xent_loss
  return logits, aux_logits


def _multiloss(logits, aux_logits, labels, num_labels, class_hist, is_training):
  max_weight = FLAGS.max_weight
  xent_loss = 0
  #main_wgt = 0.6
  if len(aux_logits) > 0:
    main_wgt = 0.7
    aux_wgt = (1 - main_wgt) / len(aux_logits)
  else:
    main_wgt = 1.0
    aux_wgt = 0
  xent_loss = main_wgt * losses.weighted_cross_entropy_loss(
      logits, labels, class_hist, max_weight=max_weight)
  for i, l in enumerate(aux_logits):
    print('loss' + str(i), ' --> ' , l)
    xent_loss += aux_wgt * losses.weighted_cross_entropy_loss(
      l, labels, class_hist, max_weight=max_weight)

  all_losses = [xent_loss]
  # get losses + regularization
  total_loss = losses.total_loss_sum(all_losses)
  if is_training:
    loss_averages_op = losses.add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
      total_loss = tf.identity(total_loss)
  return total_loss


def _dualloss(logits, mid_logits, labels, class_hist, num_labels, is_training=True):
  #loss1 = losses.cross_entropy_loss(logits, labels, weights, num_labels)
  #loss2 = losses.cross_entropy_loss(mid_logits, labels, weights, num_labels)
  #max_weight = 10
  max_weight = 1
  loss1 = losses.weighted_cross_entropy_loss(logits, labels, class_hist,
                                             max_weight=max_weight)
  loss2 = losses.weighted_cross_entropy_loss(mid_logits, labels, class_hist,
                                             max_weight=max_weight)
  #loss1 = losses.weighted_cross_entropy_loss_dense(logits, labels, weights, num_labels,
  #    max_weight=max_weight)
  #loss2 = losses.weighted_cross_entropy_loss_dense(mid_logits, labels, weights, num_labels,
  #    max_weight=max_weight)
  #wgt = 0.4
  #xent_loss = loss1 + wgt * loss2
  wgt = 0.3 # best
  #wgt = 0.2
  #wgt = 0.4
  xent_loss = (1-wgt)*loss1 + wgt*loss2
  all_losses = [xent_loss]

  # get losses + regularization
  total_loss = losses.total_loss_sum(all_losses)

  if is_training:
    loss_averages_op = losses.add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
      total_loss = tf.identity(total_loss)

  return total_loss


def minimize(loss, global_step, num_batches):
  # Calculate the learning rate schedule.
  #decay_steps = int(num_batches * FLAGS.num_epochs_per_decay)
  # Decay the learning rate exponentially based on the number of steps.
  global lr
  #base_lr = 1e-2 # for sgd
  base_lr = FLAGS.initial_learning_rate
  #TODO
  fine_lr_div = FLAGS.fine_tune_lr_factor
  #fine_lr_div = 10
  #fine_lr_div = 7
  print('LR = ', base_lr)
  print('fine_lr = LR / ', fine_lr_div)
  #lr_fine = tf.train.exponential_decay(base_lr / 10, global_step, decay_steps,
  #lr_fine = tf.train.exponential_decay(base_lr / 20, global_step, decay_steps,

  #decay_steps = int(num_batches * 30)
  #decay_steps = num_batches * FLAGS.max_epochs
  decay_steps = FLAGS.num_iters
  lr_fine = tf.train.polynomial_decay(base_lr / fine_lr_div, global_step, decay_steps,
                                      end_learning_rate=0, power=FLAGS.decay_power)
  lr = tf.train.polynomial_decay(base_lr, global_step, decay_steps,
                                 end_learning_rate=0, power=FLAGS.decay_power)
  #lr = tf.Print(lr, [lr], message='lr = ', summarize=10)

  #stairs = True
  #lr_fine = tf.train.exponential_decay(base_lr / fine_lr_div, global_step, decay_steps,
  #                                FLAGS.learning_rate_decay_factor, staircase=stairs)
  #lr = tf.train.exponential_decay(base_lr, global_step, decay_steps,
  #                                FLAGS.learning_rate_decay_factor, staircase=stairs)
  tf.summary.scalar('learning_rate', lr)
  # adam works much better here!
  if imagenet_init:
    if FLAGS.optimizer == 'adam':
      print('\nOptimizer = ADAM\n')
      opts = [tf.train.AdamOptimizer(lr_fine), tf.train.AdamOptimizer(lr)]
    elif FLAGS.optimizer == 'momentum':
      print('\nOptimizer = SGD + momentum\n')
      opts = [tf.train.MomentumOptimizer(lr_fine, 0.9), tf.train.MomentumOptimizer(lr, 0.9)]
    else:
      raise ValueError('unknown optimizer')
    return train_helper.minimize_fine_tune(opts, loss, global_step, 'head')
  else:
    opt = tf.train.AdamOptimizer(lr)
    #opt = tf.train.MomentumOptimizer(lr, 0.9)
    return train_helper.minimize(opt, loss, global_step)
  #opts = [tf.train.RMSPropOptimizer(lr_fine, momentum=0.9, centered=True),
  #        tf.train.RMSPropOptimizer(lr, momentum=0.9, centered=True)]
  #opts = [tf.train.MomentumOptimizer(lr_fine, 0.9), tf.train.MomentumOptimizer(lr, 0.9)]



def train_step(sess, run_ops):
  global train_step_iter
  if apply_jitter:
    feed_dict = _get_train_feed()
    vals = sess.run(run_ops, feed_dict=feed_dict)
  else:
    vals = sess.run(run_ops)
  train_step_iter += 1
  #img = vals[-3]
  #print(img.shape)
  ##print(img.mean())
  #for i in range(img.shape[0]):
  #  rgb = img[i]
  #  print(rgb.min())
  #  print(rgb.max())
  #  ski.io.imsave(join('/home/kivan/datasets/results/tmp/debug', str(i)+'.png'),
  #                rgb.astype(np.uint8))
  return vals


def num_batches():
  return train_dataset.num_examples() // FLAGS.batch_size


def image_size(net):
  return net.get_shape().as_list()[height_dim:height_dim+2]

def _build_dilated(image, is_training=False):
  #image = tf.Print(image, [tf.shape(image)], message='img_shape = ', summarize=10)
  bn_params['is_training'] = is_training
  with arg_scope([layers.conv2d],
      data_format=data_format, stride=1, padding='SAME', activation_fn=None,
      normalizer_fn=None, normalizer_params=None,
      weights_initializer=init_func, biases_initializer=None,
      weights_regularizer=layers.l2_regularizer(weight_decay)):
    with tf.variable_scope('conv0'):
      net = layers.conv2d(image, 2*growth, 7, stride=2)
      #net = layers.conv2d(image, 2*growth, 7, stride=1)
      # TODO
      net = tf.contrib.layers.batch_norm(net, **bn_params)
      net = tf.nn.relu(net)

    net = layers.max_pool2d(net, 2, stride=2, padding='SAME',
                            data_format=data_format, scope='pool0')

    skip_layers = []

    # no diff with double BN from orig densenet, first=True
    net = dense_block(net, block_sizes[0], growth, 'block0', is_training, first=True)
    #net, skip = dense_block(net, block_sizes[0], growth, 'block0', is_training,
    #    first=True, split=True)
    #skip_layers.append([skip, 256, growth_up, 'block0_mid_refine', depth])
    #skip_layers.append([skip, up_sizes[0], growth_up, 'block0_mid_refine'])
    skip_layers.append([net, up_sizes[0], growth_up, 'block0_refine'])
    net, _ = transition(net, compression, 'block0/transition')
    #skip_layers.append([skip, up_sizes[0], growth_up, 'block0_refine'])

    #net, skip = dense_block(net, block_sizes[1], growth, 'block1', is_training, split=True)
    #skip_layers.append([skip, up_sizes[1], growth_up, 'block1_mid_refine'])
    net = dense_block(net, block_sizes[1], growth, 'block1', is_training)
    skip_layers.append([net, up_sizes[1], growth_up, 'block1_refine'])
    net, _ = transition(net, compression, 'block1/transition')
    #skip_layers.append([skip, up_sizes[1], growth_up, 'block1_refine'])

    # works the same with split, not 100%
    #context_pool_num = 3
    #context_pool_num = 4
    context_pool_num = 5
    #net, skip = dense_block(net, block_sizes[2], growth, 'block2', is_training, split=True)
    #skip_layers.append([skip, up_sizes[2], growth_up, 'block2_mid_refine'])
    net = dense_block(net, block_sizes[2], growth, 'block2', is_training)
    #skip_layers.append([net, up_sizes[3], growth_up, 'block2_refine'])
    #skip_layers.append([net, up_sizes[2], growth_up, 'block2_refine'])
    net, _ = transition(net, compression, 'block2/transition', stride=1)

    bsz = 2
    paddings, crops = tf.required_space_to_batch_paddings(image_size(net), [bsz, bsz])
    net = tf.space_to_batch(net, paddings=paddings, block_size=bsz)
    net = dense_block(net, block_sizes[3], growth, 'block3', is_training)
    net = tf.batch_to_space(net, crops=crops, block_size=bsz)
    print('before context = ', net)

    with tf.variable_scope('head'):
      net = BNReluConv(net, 512, 'bottleneck', k=1)
      net = _pyramid_pooling(net, size=context_pool_num)
      #net = BNReluConv(net, context_size, 'context_conv', k=3)

      print('Before upsampling: ', net)

      all_logits = [net]
      for skip_layer in reversed(skip_layers):
        net = refine(net, skip_layer)
        all_logits.append(net)
        print('after upsampling = ', net)

      all_logits = [all_logits[0], all_logits[-1]]
      #all_logits = [all_logits[1], all_logits[-1]]
      #all_logits = [all_logits[2], all_logits[-1]]

  with tf.variable_scope('head'):
    for i, logits in enumerate(all_logits):
      with tf.variable_scope('logits_'+str(i)):
      # FIX
      #net = tf.nn.relu(layers.batch_norm(net, **bn_params))
      #logits = layers.conv2d(net, FLAGS.num_classes, 1, activation_fn=None,
      #                       data_format=data_format)
        logits = layers.conv2d(tf.nn.relu(logits), FLAGS.num_classes, 1,
                               activation_fn=None, data_format=data_format)

        if data_format == 'NCHW':
          logits = tf.transpose(logits, perm=[0,2,3,1])
        input_shape = tf.shape(image)[height_dim:height_dim+2]
        logits = tf.image.resize_bilinear(logits, input_shape, name='resize_logits')
        all_logits[i] = logits
    logits = all_logits.pop()
    return logits, all_logits

#def _loss(logits, labels, weights, is_training=True):
#  #TODO
#  #xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=1)
#  xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=10)
#  #xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=20)
#  #xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=50)
#  #xent_loss = losses.weighted_cross_entropy_loss(logits, labels, weights, max_weight=100)
#  all_losses = [xent_loss]
#
#  # get losses + regularization
#  total_loss = losses.total_loss_sum(all_losses)
#
#  if is_training:
#    loss_averages_op = losses.add_loss_summaries(total_loss)
#    with tf.control_dependencies([loss_averages_op]):
#      total_loss = tf.identity(total_loss)
#
#  return total_loss

