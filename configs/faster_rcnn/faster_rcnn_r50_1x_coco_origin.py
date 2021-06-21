_base_ = [
    '../_base_/models/faster_rcnn_r50_origin.py',# 模型
    '../_base_/datasets/coco_detection2.py',#数据集
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'#调度器
]
