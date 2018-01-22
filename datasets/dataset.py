import os

class Dataset(object):
  def __init__(self, data_dir, subsets):
    self.subsets = subsets
    self.filenames = []
    for subset in subsets:
      subset_dir = os.path.join(data_dir, subset)
      files = next(os.walk(subset_dir))[2]
      self.filenames.extend([os.path.join(subset_dir, f) for f in files])


  def num_classes(self):
    return self.num_classes

  def num_examples(self):
    return len(self.filenames)

  def get_filenames(self):
    return self.filenames

  #def enqueue(self, sess, enqueue_op, placeholder):
  #  for f in self.filenames:
  #    sess.run([enqueue_op], feed_dict={placeholder: f})
  #  #for i in range(10):
  #  #  sess.run([enqueue_op], feed_dict={placeholder: self.filenames[i]})
  #  #sess.run([enqueue_op], feed_dict={placeholder: self.filenames[0]})

