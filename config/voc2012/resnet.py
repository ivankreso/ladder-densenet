import os
import tensorflow as tf
import train_helper

MODEL_PATH = './models/voc2012/resnet.py'
SAVE_DIR = os.path.join('/home/kivan/datasets/results/tmp/voc2012/',
                        train_helper.get_time_string())

DATASET_DIR = '/home/kivan/datasets/VOC2012/tensorflow/'
#DATASET_DIR = '/home/kivan/datasets/voc2012_aug/tensorflow/'
IMG_HEIGHT, IMG_WIDTH = 500, 500

# SGD
# 2e-2 to large
#tf.app.flags.DEFINE_float('initial_learning_rate', 1e-2, '')
# 7 to big
tf.app.flags.DEFINE_integer('num_epochs_per_decay', 5, '')
#tf.app.flags.DEFINE_integer('num_epochs_per_decay', 6, '')
# ADAM
#tf.app.flags.DEFINE_float('initial_learning_rate', 4e-4, '')
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, '')
#tf.app.flags.DEFINE_integer('num_epochs_per_decay', 5, '')

tf.app.flags.DEFINE_integer('batch_size', 5, '')
tf.app.flags.DEFINE_integer('batch_size_valid', 3, '')
tf.app.flags.DEFINE_integer('num_validations_per_epoch', 1, '')
tf.app.flags.DEFINE_integer('max_epochs', 30, 'Number of epochs to run.')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.5,
                          """Learning rate decay factor.""")

#povecaj_lr za w=1
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', SAVE_DIR, \
    """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('resume_path', '', '')
tf.app.flags.DEFINE_integer('img_width', IMG_WIDTH, '')
tf.app.flags.DEFINE_integer('img_height', IMG_HEIGHT, '')
tf.app.flags.DEFINE_integer('img_channels', 3, '')
tf.app.flags.DEFINE_integer('max_weight', 10, '')

tf.app.flags.DEFINE_string('model_path', MODEL_PATH, '')
tf.app.flags.DEFINE_string('dataset_dir', DATASET_DIR, '')
tf.app.flags.DEFINE_string('debug_dir', os.path.join(SAVE_DIR, 'debug'), '')
tf.app.flags.DEFINE_integer('num_classes', 21, '')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
tf.app.flags.DEFINE_boolean('draw_predictions', False, 'Whether to draw.')
tf.app.flags.DEFINE_boolean('save_net', True, 'Whether to save.')
tf.app.flags.DEFINE_boolean('no_valid', False, 'Whether to save.')

tf.app.flags.DEFINE_integer('seed', 66478, '')

