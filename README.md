# detection
Microangioma Detection


# train
python tools/train.py configs/cascade_rpn/cascade_rcnn_x101_32x4d_caffe_fpn_1x_coco_rpn.py --gpus 1  --work_dir work_dirs

# test
python tools/test.py configs/cascade_rpn/cascade_rcnn_x101_32x4d_caffe_fpn_1x_coco_rpn.py work_dirs/epoch_100.pth --out ./result/result_100.pkl --eval bbox --show

# data
Customize your image path according to the following directory：

mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017

and modify configs/_base_/datasets/coco_detection3.py data_root、ann_file、img_prefix
