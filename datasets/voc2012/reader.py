import numpy as np
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


def _read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'height':         tf.FixedLenFeature([], tf.int64),
          'width':          tf.FixedLenFeature([], tf.int64),
          'channels':       tf.FixedLenFeature([], tf.int64),
          'num_labels':     tf.FixedLenFeature([], tf.int64),
          'img_name':       tf.FixedLenFeature([], tf.string),
          'img':            tf.FixedLenFeature([], tf.string),
          'labels':         tf.FixedLenFeature([], tf.string),
          #'label_weights':  tf.FixedLenFeature([], tf.string),
          'class_hist':  tf.FixedLenFeature([], tf.string),
      })

  image = tf.to_float(tf.decode_raw(features['img'], tf.uint8, name='decode_image'))
  labels = tf.to_int32(tf.decode_raw(features['labels'], tf.int8, name='decode_labels'))
  #weights = tf.decode_raw(features['label_weights'], tf.float32, name='decode_weights')
  class_hist = tf.decode_raw(features['class_hist'], tf.int32, name='decode_class_hist')
  img_name = features['img_name']
  num_labels = tf.to_float(features['num_labels'])
  image = tf.reshape(image, shape=[FLAGS.img_height, FLAGS.img_width, FLAGS.img_channels])
  labels = tf.reshape(labels, shape=[FLAGS.img_height, FLAGS.img_width, 1])
  #weights = tf.reshape(weights, shape=[FLAGS.img_height, FLAGS.img_width, 1])
  class_hist = tf.reshape(class_hist, shape=[FLAGS.num_classes])

  return image, labels, num_labels, class_hist, img_name


def num_examples(dataset):
  return int(dataset.num_examples() // FLAGS.batch_size)


def inputs(dataset, is_training=False, num_epochs=None):
  """Reads input data num_epochs times.
  Args:
    dataset:
    num_epochs: Number of times to read the input data, or 0/None to train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
    * labels is an int32 tensor with shape [batch_size] with the true label
  """
  shuffle = is_training
  if is_training:
    batch_size = FLAGS.batch_size
  else:
    batch_size = FLAGS.batch_size_valid
    assert dataset.num_examples() % batch_size == 0

  with tf.name_scope('input'), tf.device('/cpu:0'):
    filename_queue = tf.train.string_input_producer(dataset.get_filenames(),
        num_epochs=num_epochs, shuffle=shuffle, seed=FLAGS.seed,
        capacity=dataset.num_examples())

    image, labels, num_labels, class_hist, img_name = _read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # Run this in two threads to avoid being a bottleneck.
    image, labels, num_labels, class_hist, img_name = tf.train.batch(
        [image, labels, num_labels, class_hist, img_name], batch_size=batch_size,
        num_threads=2, capacity=64)

    return image, labels, num_labels, class_hist, img_name
