# detection
Microangioma Detection


# train
python tools/train.py configs/cascade_rpn/cascade_rcnn_x101_32x4d_caffe_fpn_1x_coco_rpn.py --gpus 1  --work_dir work_dirs

# test
python tools/test.py configs/cascade_rpn/cascade_rcnn_x101_32x4d_caffe_fpn_1x_coco_rpn.py work_dirs/epoch_100.pth --out ./result/result_100.pkl --eval bbox --show

# data
Customize your image path according to the following directory：
![image](https://user-images.githubusercontent.com/54968734/122735489-a9845b00-d2b1-11eb-812f-dd8353db50d8.png)

and modify configs/_base_/datasets/coco_detection3.py data_root、ann_file、img_prefix
