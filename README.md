
## Dataset preparation

1. Add this repository to your PYTHONPATH variable:
```
PYTHONPATH=$PYTHONPATH:/path/to/fork
```

2. Run script:
- for Cityscapes:
Download the dataset [here](https://www.cityscapes-dataset.com/downloads).

```
cd datasets/cityscapes
python prepare_dataset.py --data_dir=/path/to/data --gt_dir=/path/to/labels --save_dir=/path/to/save
```

- for PASCAL VOC2012:
Download the dataset [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012).

```
cd datasets/voc2012
python prepare_dataset.py --data_dir=/path/to/data --save_dir=/path/to/save
```

## Training

- for Cityscapes:
```
python train.py --config_path='/path/to/fork/config/cityscapes/densenet.py' --batch_size=5 --num_iters=3000
```

- for PASCAL VOC2012:
```
python train.py --config_path='/path/to/fork/config/voc2012/densenet.py' --batch_size=5 --num_iters=3000
```

## Evaluation

For evaluation please refer to the new version of the repository:

https://github.com/ivankreso/LDN
