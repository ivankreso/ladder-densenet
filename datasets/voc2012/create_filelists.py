import os
from os.path import join

def dump_to_file(path, lines):
  fp = open(path, 'w')
  for line in lines:
    fp.write(line+'\n')

path = '/home/kivan/datasets/voc2012_aug/config.txt'
fp = open(path)

lines = fp.readlines()
train = []
val = []
for line in lines:
  split = line.strip().split()
  if split[1] == '1':
    train.append(split[0])
  elif split[1] == '0':
    val.append(split[0])
  else:
    raise 1

print('train = ', len(train))
print('val = ', len(val))

root_dir = '/home/kivan/datasets/voc2012_aug/'

dump_to_file(join(root_dir, 'train.txt'), train)
dump_to_file(join(root_dir, 'val.txt'), val)
