import os
from os.path import join

class Dataset(object):
  class_info = [[128,64,128,   'background'],
                [244,35,232,   'aeroplane'],
                [70,70,70,     'bicycle'],
                [102,102,156,  'bird'],
                [190,153,153,  'boat'],
                [153,153,153,  'bottle'],
                [250,170,30,   'bus'],
                [220,220,0,    'car'],
                [107,142,35,   'cat'],
                [152,251,152,  'chair'],
                [70,130,180,   'cow'],
                [220,20,60,    'diningtable'],
                [255,0,0,      'dog'],
                [0,0,142,      'horse'],
                [0,0,70,       'motorbike'],
                [0,60,100,     'person'],
                [0,80,100,     'potted plant'],
                [0,0,230,      'sheep'],
                [0,0,230,      'sofa'],
                [0,0,230,      'train'],
                [119,11,32,    'monitor']]

  def __init__(self, data_dir, subset_path, subset):
    self.subset = subset
    fp = open(subset_path)
    filenames = [line.strip() + '.tfrecords' for line in fp.readlines()]
    #print(filenames)
    self.data_dir = data_dir
    self.filenames = [join(data_dir, f) for f in filenames]

  def num_classes(self):
    return self.num_classes

  def num_examples(self):
    return len(self.filenames)

  def get_filenames(self):
    return self.filenames

