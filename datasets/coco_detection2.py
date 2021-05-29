dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
data_root = '/public/home/yuqi/shijiarui/mmdetection/data/'
# data_root = '/public/home/yuqi/tile_competition/SliceData/tile/'
# data_root = '/public/home/yuqi/tile_competition/tcdata/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(
    mean=[134.82, 79.32, 47.61], std=[29.19, 15.07, 10.10], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    # dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False),
    # dict(type='Resize', img_scale=(8192, 6000), keep_ratio=False),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        # img_scale=(1024, 1024),
        # img_scale=(8192, 6000),
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]

        )
]
data = dict(
    samples_per_gpu=2,#gpu数目 2? 在schedule中修改lr，batch_size = gpu数量 * 每个gpu读取的图片数
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train_crop_0325.json',
        # ann_file=data_root + 'annotations/instances_train.json',
        #img_prefix=data_root + 'train_clahe/',
        img_prefix=data_root + 'train_crop/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val_crop0314.json',
        # ann_file=data_root + 'annotations/instances_val.json',
        #img_prefix=data_root + 'val_clahe/',
        img_prefix=data_root + 'val_crop/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_test_en.json',
        # img_prefix=data_root + 'test1/',
        ann_file=data_root + 'annotations/instances_val_crop_0404.json',
        # ann_file=data_root + 'annotations/instances_testB.json',
        img_prefix=data_root + 'test_val/',
        
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')