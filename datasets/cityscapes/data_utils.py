import numpy as np
from cityscapes import CityscapesDataset


def convert_ids(img, has_hood=True, ignore_id=255):
  img_train = np.zeros_like(img)
  img_train.fill(ignore_id)
  car_mask = img == 1
  height = car_mask.shape[0]
  if has_hood:
    car_mask[height-5:,...] = True
  for i, cid in enumerate(CityscapesDataset.train_ids):
    img_train[img==cid] = i
  return img_train, car_mask

def get_class_hist(gt_img, num_classes):
  hist = np.zeros(num_classes, dtype=np.int32)
  #hist = np.ones(num_classes, dtype=np.int32)
  #num_labels = (gt_img >= 0).sum()
  for i in range(num_classes):
    mask = gt_img == i
    hist[i] += mask.sum()
  num_labels = (gt_img < num_classes).sum()
  return hist, num_labels

def get_class_weights_old(gt_img, num_classes=19, max_wgt=100):
  height = gt_img.shape[0]
  width = gt_img.shape[1]
  weights = np.zeros((height, width), dtype=np.float32)
  num_labels = (gt_img >= 0).sum()
  for i in range(num_classes):
    mask = gt_img == i
    class_cnt = mask.sum()
    if class_cnt > 0:
      wgt = min(max_wgt, 1.0 / (class_cnt / num_labels))
      weights[mask] = wgt
      #print(i, wgt)
  return weights, num_labels
