import os
from os.path import join
import sys
import time
from shutil import copyfile
from tqdm import trange

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import utils

np.set_printoptions(linewidth=250)

tf.app.flags.DEFINE_string('config_path', '', """Path to experiment config.""")
FLAGS = tf.app.flags.FLAGS

utils.import_module('config', FLAGS.config_path)
print(FLAGS.config_path)


def train(model):
  """ Trains the network
    Args:
      model: module containing model architecture
  """
  config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
  #config.operation_timeout_in_ms = 5000   # terminate on long hangs
  #sess = tf.Session(config=config)
  with tf.Session(config=config) as sess:
    if FLAGS.seed >= 0:
      tf.set_random_seed(FLAGS.seed)
      np.random.seed(FLAGS.seed)

    # Build a Graph that computes the logits predictions from the inference model.
    train_ops, init_op, init_feed = model.build('train')
    num_params = utils.get_num_params()
    vars_to_restore = tf.contrib.framework.get_variables_to_restore()
    if FLAGS.no_valid is False:
      valid_ops = model.build('validation')
    loss = train_ops[0]

    num_batches = model.num_batches()
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    # TODO
    train_op = model.minimize(loss, global_step, num_batches)
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):
    #  train_op = model.minimize(loss, global_step, num_batches)

    print('\nNumber of parameters = ', num_params)
    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_num_epochs)

    #sess.run(tf.global_variables_initializer())
    #sess.run(tf.local_variables_initializer())
    #if init_op != None:
    #  print('\nInitializing pretrained weights...')
    #  sess.run(init_op, feed_dict=init_feed)

    #if len(FLAGS.resume_path) > 0:
    #  print('\nResuming training from:', FLAGS.resume_path)
    #  assert tf.gfile.Exists(FLAGS.resume_path)
    #  resnet_restore = tf.train.Saver(model.variables_to_restore())
    #  resnet_restore.restore(sess, FLAGS.resume_path)

    sess.run(tf.global_variables_initializer())
    if len(FLAGS.resume_path) > 0:
      print(f'\nRestoring params from: {FLAGS.resume_path}\n')
      #print(tf.train.latest_checkpoint(FLAGS.resume_path))
      #assert tf.gfile.Exists(FLAGS.resume_path)
      resnet_restore = tf.train.Saver(vars_to_restore)
      resnet_restore.restore(sess, FLAGS.resume_path)
    elif init_op != None:
      print('\nInitializing from pretrained weights...')
      sess.run(init_op, feed_dict=init_feed)
    else:
      print('All params are using random init')
    sess.run(tf.local_variables_initializer())

    # Build the summary operation based on the TF collection of Summaries.
    #summary_op = tf.merge_all_summaries()
    summary_op = tf.summary.merge_all()

    # Start the queue runners.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph=sess.graph)
    #TODO tf.summary.FileWriter()

    #init_vars = utils.get_variables(sess)
    #utils.print_variable_diff(sess, init_vars)
    #variable_map = utils.get_variable_map()
    # take the train loss moving average
    #loss_avg_train = variable_map['total_loss/avg:0']
    print('Training network...\nModel saving =', FLAGS.save_net)
    train_loss_val = 0
    train_data, valid_data = model.init_eval_data()
    ex_start_time = time.perf_counter()
    iter_num = 0
    for epoch_num in range(1, FLAGS.max_num_epochs + 1):
      print('\nnvim ' + FLAGS.train_dir + 'model.py')
      print('tensorboard --logdir=' + FLAGS.train_dir + '\n')
      #num_batches = model.num_batches() // FLAGS.num_validations_per_epoch
      model.start_epoch(train_data)
      #for step in range(0):
      duration = 0
      for step in range(num_batches):
        #if iter_num >= FLAGS.num_iters:
        #  break
        iter_num += 1
        run_ops = train_ops + [train_op, global_step]
        #run_ops = [train_op, loss, logits, labels, draw_data, img_name, global_step]
        start_time = time.perf_counter()        
        if False:
        #if step % 400 == 0:
          run_ops += [summary_op]
          #run_ops += [summary_op]
          loss_val = ret_val[0]
          summary_str = ret_val[-1]
          global_step_val = ret_val[-2]
          summary_writer.add_summary(summary_str, global_step_val)
        else:
          #ret_val = sess.run(run_ops, feed_dict=feed_dict)
          ret_val = model.train_step(sess, run_ops)
          #if step % 100 == 0:
          #  model.evaluate_output(ret_val, step)
          #utils.print_grad_stats(grads_val, grad_tensors)
          #if step > 20:
          #  run_metadata = tf.RunMetadata()
          #  ret_val = sess.run(run_ops, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
          #                     run_metadata=run_metadata)
          #  trace = timeline.Timeline(step_stats=run_metadata.step_stats)
          #  trace_file = open('timeline.ctf.json', 'w')
          #  trace_file.write(trace.generate_chrome_trace_format())
          #  raise 1

        #img_prefix = img_prefix[0].decode("utf-8")

        #if FLAGS.draw_predictions and step % 50 == 0:// Controls the font size in pixels
        #  model.draw_prediction('train', epoch_num, step, ret_val)

        if step % 20 == 0:
          duration = time.perf_counter() - start_time
          examples_per_sec = FLAGS.batch_size / duration
          loss_val = ret_val[0]
          format_str = '%s: epoch %03d / %03d, step %04d / %04d, iter %06d / %06d, loss = %.2f \
            (%.1f examples/sec)'
          #print('lr = ', clr)
          num_iters = FLAGS.max_num_epochs * model.num_batches()
          print(format_str % (utils.get_expired_time(ex_start_time), epoch_num,
                              FLAGS.max_num_epochs,
                              step, model.num_batches(), iter_num, num_iters,
                              loss_val, examples_per_sec))

        assert not np.isnan(loss_val), 'Model diverged with loss = NaN'
        # estimate training accuracy on the last 40% of the epoch
        if step > int(0.5 * num_batches):
          model.update_stats(ret_val)

      is_best = model.end_epoch(train_data)
      #utils.print_variable_diff(sess, init_vars)
      if FLAGS.no_valid is False:
        is_best = model.evaluate('valid', sess, epoch_num, valid_ops, valid_data)

      model.print_results(train_data, valid_data)
      model.plot_results(train_data, valid_data)
      #  eval_helper.plot_training_progress(os.path.join(FLAGS.train_dir, 'stats'), plot_data)

      # Save the best model checkpoint
      if FLAGS.save_net and is_best:
        print('Saving model...')
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        #saver.save(sess, checkpoint_path, global_step=epoch_num)
        saver.save(sess, checkpoint_path)
      elif not FLAGS.save_net:
        print('WARNING: not saving...')
      #if iter_num >= FLAGS.num_iters:
      #  break

    coord.request_stop()
    coord.join(threads)
    #sess.close()


def main(argv=None):  # pylint: disable=unused-argument
  model = utils.import_module('model', FLAGS.model_path)

  if tf.gfile.Exists(FLAGS.train_dir):
    raise ValueError('Train dir exists: ' + FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)

  stats_dir = join(FLAGS.train_dir, 'stats')
  tf.gfile.MakeDirs(stats_dir)
  tf.gfile.MakeDirs(join(FLAGS.debug_dir, 'train'))
  tf.gfile.MakeDirs(join(FLAGS.debug_dir, 'valid'))
  tf.gfile.MakeDirs(join(FLAGS.train_dir, 'results'))
  f = open(join(stats_dir, 'log.txt'), 'w')
  sys.stdout = utils.Logger(sys.stdout, f)

  copyfile(FLAGS.model_path, os.path.join(FLAGS.train_dir, 'model.py'))
  copyfile(FLAGS.config_path, os.path.join(FLAGS.train_dir, 'config.py'))

  print('Experiment dir: ' + FLAGS.train_dir)
  train(model)


if __name__ == '__main__':
  tf.app.run()

