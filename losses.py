import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def add_loss_summaries(total_loss):
  """Add summaries for losses in model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.

  for l in losses + [total_loss]:
    #print(l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    #tf.scalar_summary(l.op.name + ' (raw)', l)
    #tf.scalar_summary(l.op.name, loss_averages.average(l))
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))
    #tf.scalar_summary([l.op.name + ' (raw)'], l)
    #tf.scalar_summary([l.op.name], loss_averages.average(l))

  return loss_averages_op


def total_loss_sum(losses):
  # Assemble all of the losses for the current tower only.
  #print(losses)
  # Calculate the total loss for the current tower.
  #regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  #regularization_losses = tf.contrib.losses.get_regularization_losses()
  regularization_losses = tf.losses.get_regularization_losses()
  #total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
  total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
  return total_loss


def cross_entropy_loss(logits, labels):
  print('loss: cross-entropy')
  num_pixels = -1
  with tf.name_scope(None, 'CrossEntropyLoss', [logits, labels]):
    labels = tf.reshape(labels, shape=[num_pixels])
    logits = tf.reshape(logits, [num_pixels, FLAGS.num_classes])
    mask = labels < FLAGS.num_classes
    idx = tf.where(mask)
    # # labels = tf.reshape(labels, shape=[num_pixels])
    # print(idx)
    labels = tf.to_float(labels)
    labels = tf.gather_nd(labels, idx)
    # labels = tf.boolean_mask(labels, mask)
    labels = tf.to_int32(labels)
    logits = tf.gather_nd(logits, idx)
    # logits = tf.boolean_mask(logits, mask)

    
    onehot_labels = tf.one_hot(labels, FLAGS.num_classes)
    xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels)
    
    # range_idx = tf.range(tf.shape(labels)[0], dtype=tf.int32)
    # print(range_idx, labels)
    # labels = tf.reshape(labels, shape=[-1,1])
    # range_idx = tf.reshape(range_idx, shape=[-1,1])
    # idx = tf.concat([range_idx, labels], axis=1)
    # print(idx)
    # probs = tf.nn.softmax(logits)
    # probs = tf.gather_nd(probs, idx)
    # print(probs)
    # xent = tf.square(1 - probs) * xent

    # xent = tf.pow(1 - probs, 3) * xent
    # xent = (1 - probs) * xent

    #num_labels = tf.to_float(tf.reduce_sum(num_labels))
    #num_labels = tf.reduce_sum(tf.to_float(num_labels))

    #class_hist = tf.Print(class_hist, [class_hist], 'hist = ', summarize=30)
    #num_labels = tf.reduce_sum(onehot_labels)

    #class_hist = tf.to_float(tf.reduce_sum(class_hist, axis=0))
    ##num_labels = tf.Print(num_labels, [num_labels, tf.reduce_sum(onehot_labels)], 'lab = ', summarize=30)
    #class_weights = num_labels / (class_hist + 1)
    ##class_weights = tf.Print(class_weights, [class_weights], 'wgt hist = ', summarize=30)
    ## we need to append 0 here for ignore pixels
    #class_weights = tf.concat([class_weights, [0]], axis=0)
    ##class_weights = tf.Print(class_weights, [class_weights], 'wgt hist = ', summarize=30)
    #class_weights = tf.minimum(tf.to_float(max_weight), class_weights)

    # class_weights = tf.ones([FLAGS.num_classes])
    # class_weights = tf.concat([class_weights, [0]], axis=0)
    # #class_weights = tf.Print(class_weights, [class_weights], 'wgt hist = ', summarize=30)
    # weights = tf.gather(class_weights, labels)

    xent = tf.reduce_mean(xent)
    return xent

def weighted_cross_entropy_loss(logits, labels, class_hist=None, max_weight=1):
  print('loss: cross-entropy')
  print('Using balanced loss with max weight = ', max_weight)
  num_pixels = -1
  with tf.name_scope(None, 'CrossEntropyLoss', [logits, labels]):
    labels = tf.reshape(labels, shape=[num_pixels])
    onehot_labels = tf.one_hot(labels, FLAGS.num_classes)
    logits = tf.reshape(logits, [num_pixels, FLAGS.num_classes])
    xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels)

    #num_labels = tf.to_float(tf.reduce_sum(num_labels))
    #num_labels = tf.reduce_sum(tf.to_float(num_labels))

    #class_hist = tf.Print(class_hist, [class_hist], 'hist = ', summarize=30)
    num_labels = tf.reduce_sum(onehot_labels)

    #class_hist = tf.to_float(tf.reduce_sum(class_hist, axis=0))
    ##num_labels = tf.Print(num_labels, [num_labels, tf.reduce_sum(onehot_labels)], 'lab = ', summarize=30)
    #class_weights = num_labels / (class_hist + 1)
    ##class_weights = tf.Print(class_weights, [class_weights], 'wgt hist = ', summarize=30)
    ## we need to append 0 here for ignore pixels
    #class_weights = tf.concat([class_weights, [0]], axis=0)
    ##class_weights = tf.Print(class_weights, [class_weights], 'wgt hist = ', summarize=30)
    #class_weights = tf.minimum(tf.to_float(max_weight), class_weights)

    class_weights = tf.ones([FLAGS.num_classes])
    class_weights = tf.concat([class_weights, [0]], axis=0)
    #class_weights = tf.Print(class_weights, [class_weights], 'wgt hist = ', summarize=30)
    weights = tf.gather(class_weights, labels)

    if max_weight > 1:
      raise ValueError()
      wgt_sum = tf.reduce_sum(weights)
      norm_factor = num_labels / wgt_sum
      # weights need to sum to 1
      weights = tf.multiply(weights, norm_factor)

    xent = tf.multiply(weights, xent)
    #num_labels = tf.Print(num_labels, [num_labels, wgt_sum], 'num_labels = ')
    #xent = tf.Print(xent, [xent], 'num_labels = ')
    xent = tf.reduce_sum(xent) / num_labels
    return xent


def weighted_cross_entropy_loss_dense(logits, labels, weights=None,
                                      num_labels=None, max_weight=100):
  print('loss: cross-entropy')
  num_pixels = -1
  with tf.name_scope(None, 'CrossEntropyLoss', [logits, labels]):
    labels = tf.reshape(labels, shape=[num_pixels])
    onehot_labels = tf.one_hot(labels, FLAGS.num_classes)
    logits = tf.reshape(logits, [num_pixels, FLAGS.num_classes])
    xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels)

    if num_labels is None:
      num_labels = tf.reduce_sum(onehot_labels)
    else:
      num_labels = tf.reduce_sum(num_labels)

    print('Using balanced loss with max weight = ', max_weight)
    weights = tf.reshape(weights, shape=[num_pixels])
    weights = tf.minimum(tf.to_float(max_weight), weights)
    wgt_sum = tf.reduce_sum(weights)
    norm_factor = num_labels / wgt_sum
    # weights need to sum to 1
    weights = tf.multiply(weights, norm_factor)
    xent = tf.multiply(weights, xent)

    #num_labels = tf.Print(num_labels, [num_labels, wgt_sum], 'num_labels = ')
    #xent = tf.Print(xent, [xent], 'num_labels = ')
    xent = tf.reduce_sum(xent) / num_labels
    print(xent)
    return xent


def cross_entropy_loss_old(logits, labels, weights, num_labels):
  print('loss: cross-entropy')
  num_pixels = -1
  with tf.name_scope(None, 'CrossEntropyLoss', [logits, labels, num_labels]):
    labels = tf.reshape(labels, shape=[num_pixels])
    onehot_labels = tf.one_hot(labels, FLAGS.num_classes)
    logits = tf.reshape(logits, [num_pixels, FLAGS.num_classes])
    xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels)
    weights = tf.reshape(weights, shape=[num_pixels])
    xent = tf.multiply(weights, xent)
    xent = tf.reduce_sum(xent) / tf.reduce_sum(num_labels)
    print(xent)
    return xent


def mse(yp, yt):
  num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
  with tf.name_scope('MeanSquareError'):
    yt = tf.reshape(yt, shape=[num_examples])
    yp = tf.reshape(yp, shape=[num_examples])
    return tf.reduce_mean(tf.square(yt - yp))



def weighted_cross_entropy_loss_deprecated(logits, labels, weights=None, max_weight=100):
#def weighted_cross_entropy_loss(logits, labels, weights=None, max_weight=1e2):
#def weighted_cross_entropy_loss(logits, labels, weights=None, max_weight=1e3):
  print('loss: Weighted Cross Entropy Loss')
  shape = labels.get_shape().as_list()
  print(shape)
  #num_examples = shape[0] * shape[1]
  num_examples = -1
  #num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
  with tf.name_scope(None, 'WeightedCrossEntropyLoss', [logits, labels]):
    labels = tf.reshape(labels, shape=[num_examples])
    #num_labels = tf.to_float(tf.reduce_sum(num_labels))
    one_hot_labels = tf.one_hot(tf.to_int64(labels), FLAGS.num_classes, 1, 0)
    one_hot_labels = tf.reshape(one_hot_labels, [num_examples, FLAGS.num_classes])
    num_labels = tf.to_float(tf.reduce_sum(one_hot_labels))
    logits_1d = tf.reshape(logits, [num_examples, FLAGS.num_classes])
    # todo
    #log_softmax = tf.log(tf.nn.softmax(logits_1d)) - never do this!
    log_softmax = tf.nn.log_softmax(logits_1d)
    xent = tf.reduce_sum(-tf.multiply(tf.to_float(one_hot_labels), log_softmax), 1)
    #weighted_xent = tf.mul(weights, xent)
    if weights != None:
      weights = tf.reshape(weights, shape=[num_examples])
      xent = tf.mul(tf.minimum(tf.to_float(max_weight), weights), xent)
    #weighted_xent = xent

    total_loss = tf.div(tf.reduce_sum(xent), tf.to_float(num_labels), name='value')
    print(total_loss)
    return total_loss


def flip_xent_loss(logits, labels, weights, max_weight=10):
  print('Loss: Weighted Cross Entropy Loss')
  assert(FLAGS.batch_size == 2)
  num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
  labels = tf.reshape(labels, shape=[num_examples])
  weights = tf.reshape(weights, shape=[num_examples])
  #num_labels = tf.to_float(tf.reduce_sum(num_labels))
  with tf.name_scope('FlipXentLoss', [logits, labels]):
    one_hot_labels = tf.one_hot(tf.to_int64(labels), FLAGS.num_classes, 1, 0)
    one_hot_labels = tf.reshape(one_hot_labels, [num_examples, FLAGS.num_classes])
    num_labels = tf.to_float(tf.reduce_sum(one_hot_labels))
    #print(logits[].get_shape())
    logits_1d = tf.reshape(logits, [num_examples, FLAGS.num_classes])
    # TODO
    #log_softmax = tf.log(tf.nn.softmax(logits_1d))
    log_softmax = tf.nn.log_softmax(logits_1d)
    xent = tf.reduce_sum(tf.mul(tf.to_float(one_hot_labels), log_softmax), 1)
    #weighted_xent = tf.mul(weights, xent)
    weighted_xent = tf.mul(tf.minimum(tf.to_float(max_weight), weights), xent)
    #weighted_xent = xent

    total_loss = - tf.div(tf.reduce_sum(weighted_xent), num_labels, name='value')
    return total_loss


def multiclass_hinge_loss(logits, labels, weights):
  print('loss: Hinge loss')
  num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
  num_classes = FLAGS.num_classes
  with tf.op_scope([logits, labels], None, 'MulticlassHingeLoss'):
    #logits = tf.reshape(logits, [num_examples, num_classes])
    #labels = tf.reshape(labels, [num_examples])
    #weights = tf.reshape(weights, [num_examples])
    logits = tf.reshape(logits, [-1, num_classes])
    labels = tf.reshape(labels, [-1])
    weights = tf.reshape(weights, [-1])
    select_mask = tf.greater_equal(labels, 0)
    logits = tf.boolean_mask(logits, select_mask)
    labels = tf.boolean_mask(labels, select_mask)
    weights = tf.boolean_mask(weights, select_mask)
    num_examples = tf.reduce_sum(tf.to_int32(select_mask))
    #num_examples = tf.Print(num_examples, [num_examples, num_labels_old], 'num_examples = ')
    #print(labels)
    #print(logits)
    #print(weights)
    #print(select_mask)
    partitions = tf.one_hot(tf.to_int64(labels), FLAGS.num_classes, 1, 0, dtype=tf.int32)
    #print(partitions)
    #one_hot_labels = tf.one_hot(tf.to_int64(labels), FLAGS.num_classes, 1, 0)
    #one_hot_labels = tf.reshape(one_hot_labels, [num_examples, FLAGS.num_classes])
    #partitions = tf.to_int32(one_hot_labels)

    num_partitions = 2
    scores, score_yt = tf.dynamic_partition(logits, partitions, num_partitions)
    #scores = tf.reshape(scores, [num_examples, num_classes - 1])
    #score_yt = tf.reshape(score_yt, [num_examples,  1])
    scores = tf.reshape(scores, [-1, num_classes - 1])
    score_yt = tf.reshape(score_yt, [-1,  1])
    #print(scores)
    #print(score_yt)

    #hinge_loss = tf.maximum(0.0, scores - score_yt + margin)
    hinge_loss = tf.square(tf.maximum(0.0, scores - score_yt + 1.0))
    hinge_loss = tf.reduce_sum(hinge_loss, 1)

    #total_loss = tf.reduce_sum(tf.mul(weights, hinge_loss))
    #total_loss = tf.div(total_loss, tf.to_float(num_examples), name='value')
    total_loss = tf.reduce_mean(tf.mul(tf.minimum(100.0, weights), hinge_loss))

    #tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)
    #tf.nn.l2_loss(t, name=None)
    return total_loss


# def metric_hinge_loss(logits, labels, weights, num_labels):
#   print('loss: Hinge loss')
#   num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
#   with tf.op_scope([logits, labels], None, 'weightedhingeloss'):
#     one_hot_labels = tf.one_hot(tf.to_int64(labels), FLAGS.num_classes, 1, 0)
#     logits_1d = tf.reshape(logits, [num_examples, FLAGS.num_classes])
#     #codes = tf.nn.softmax(logits_1d)
#     codes = tf.nn.l2_normalize(logits_1d, 1)
#     # works worse
#     # l2 loss -> bad!
#     # todo - this is not true svm loss, try it from cs231n
#     l2_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.to_float(one_hot_labels) - codes), 1))
#     m = 0.2
#     #l2_dist = tf.reduce_sum(tf.square(tf.to_float(one_hot_labels) - codes), 1)
#     #m = 0.2 ** 2
#     #m = 0.1 ** 2
#     #m = 0.3 ** 2
#     for i in range(num_classes):
#       for j in range(num_classes):
#         raise valueerror(1)
#     hinge_loss = tf.maximum(tf.to_float(0), l2_dist - m)
#     total_loss = tf.reduce_sum(tf.mul(weights, hinge_loss))

#     total_loss = tf.div(total_loss, tf.to_float(num_labels), name='value')
#     tf.add_to_collection(slim.losses.LOSSES_COLLECTION, total_loss)

#     #tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)
#     #tf.nn.l2_loss(t, name=None)
#     return total_loss

#def weighted_hinge_loss(logits, labels, weights, num_labels):
#  print('Loss: Hinge Loss')
#  num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
#  with tf.op_scope([logits, labels], None, 'WeightedHingeLoss'):
#    weights = tf.reshape(weights, shape=[num_examples])
#    labels = tf.reshape(labels, shape=[num_examples])
#    num_labels = tf.to_float(tf.reduce_sum(num_labels))
#    one_hot_labels = tf.one_hot(tf.to_int64(labels), FLAGS.num_classes, 1, 0)
#    one_hot_labels = tf.reshape(one_hot_labels, [num_examples, FLAGS.num_classes])
#    logits_1d = tf.reshape(logits, [num_examples, FLAGS.num_classes])
#    #codes = tf.nn.softmax(logits_1d)
#    codes = tf.nn.l2_normalize(logits_1d, 1)
#    # works worse
#    #l2_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.to_float(one_hot_labels) - codes), 1))
#    #m = 0.2
#    l2_dist = tf.reduce_sum(tf.square(tf.to_float(one_hot_labels) - codes), 1)
#    m = 0.2 ** 2
#    #m = 0.1 ** 2
#    #m = 0.3 ** 2
#    hinge_loss = tf.maximum(tf.to_float(0), l2_dist - m)
#    total_loss = tf.reduce_sum(tf.mul(weights, hinge_loss))
#
#    total_loss = tf.div(total_loss, tf.to_float(num_labels), name='value')
#    tf.add_to_collection(slim.losses.LOSSES_COLLECTION, total_loss)
#
#    #tf.nn.l2_normalize(x, dim, epsilon=1e-12, name=None)
#    #tf.nn.l2_loss(t, name=None)
#    return total_loss

# def flip_xent_loss_symmetric(logits, labels, weights, num_labels):
#   print('Loss: Weighted Cross Entropy Loss')
#   num_examples = FLAGS.img_height * FLAGS.img_width
#   with tf.op_scope([logits, labels], None, 'WeightedCrossEntropyLoss'):
#     labels = tf.reshape(labels, shape=[2, num_examples])
#     weights = tf.reshape(weights, shape=[2, num_examples])
#     num_labels = tf.to_float(tf.reduce_sum(num_labels))
#     #num_labels = tf.to_float(num_labels[0])
#     logits_flip = logits[1,:,:,:]
#     #weights_flip = weights[1,:]

#     logits = logits[0,:,:,:]
#     weights = weights[0,:]
#     labels = labels[0,:]
#     one_hot_labels = tf.one_hot(tf.to_int64(labels), FLAGS.num_classes, 1, 0)
#     one_hot_labels = tf.reshape(one_hot_labels, [num_examples, FLAGS.num_classes])

#     #logits_orig, logits_flip = tf.split(0, 2, logits)
#     logits_flip = tf.image.flip_left_right(logits_flip)
#     #print(logits[].get_shape())
#     logits_1d = tf.reshape(logits, [num_examples, FLAGS.num_classes])
#     logits_1d_flip = tf.reshape(logits_flip, [num_examples, FLAGS.num_classes])
#     # TODO
#     log_softmax = tf.nn.log_softmax(logits_1d)

#     #log_softmax_flip = tf.nn.log_softmax(logits_1d_flip)
#     softmax_flip = tf.nn.softmax(logits_1d_flip)
#     xent = tf.reduce_sum(tf.mul(tf.to_float(one_hot_labels), log_softmax), 1)
#     weighted_xent = tf.mul(tf.minimum(tf.to_float(100), weights), xent)
#     xent_flip = tf.reduce_sum(tf.mul(softmax_flip, log_softmax), 1)
#     xent_flip = tf.mul(tf.minimum(tf.to_float(100), weights), xent_flip)
#     #weighted_xent = tf.mul(weights, xent)
#     #weighted_xent = xent

#     #total_loss = tf.div(- tf.reduce_sum(weighted_xent_flip),
#     #                    num_labels, name='value')
#     total_loss = - tf.div(tf.reduce_sum(weighted_xent) + tf.reduce_sum(xent_flip),
#                           num_labels, name='value')

#     tf.add_to_collection(slim.losses.LOSSES_COLLECTION, total_loss)
#     return total_loss

