# _base_ = [
#     '../_base_/datasets/coco_detection.py',
#     '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
# ]

# rpn_weight = 0.7
# model = dict(
#     type='CascadeRCNN',
#     pretrained='open-mmlab://resnext101_32x4d',
#     backbone=dict(
#         type='ResNeXt',
#         depth=101,
#         groups=32,
#         base_width=4,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         style='pytorch',
#         dcn=dict( #在最后三个block加入可变形卷积 
#             modulated=False, deformable_groups=1, fallback_on_stride=False),
#             stage_with_dcn=(False, True, True, True)),
#     neck=dict(
#         type='FPN',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         num_outs=5),
#     rpn_head=dict(
#         _delete_=True,
#         type='CascadeRPNHead',
#         num_stages=2,
#         stages=[
#             dict(
#                 type='StageCascadeRPNHead',
#                 in_channels=256,
#                 feat_channels=256,
#                 anchor_generator=dict(
#                     type='AnchorGenerator',
#                     scales=[8],
#                     ratios=[1.0],
#                     strides=[4, 8, 16, 32, 64]),
#                 adapt_cfg=dict(type='dilation', dilation=3),
#                 bridged_feature=True,
#                 sampling=False,
#                 with_cls=False,
#                 reg_decoded_bbox=True,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=(.0, .0, .0, .0),
#                     target_stds=(0.1, 0.1, 0.5, 0.5)),
#                 loss_bbox=dict(
#                     type='IoULoss', linear=True,
#                     loss_weight=10.0 * rpn_weight)),
#             dict(
#                 type='StageCascadeRPNHead',
#                 in_channels=256,
#                 feat_channels=256,
#                 adapt_cfg=dict(type='offset'),
#                 bridged_feature=False,
#                 sampling=True,
#                 with_cls=True,
#                 reg_decoded_bbox=True,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=(.0, .0, .0, .0),
#                     target_stds=(0.05, 0.05, 0.1, 0.1)),
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=True,
#                     loss_weight=1.0 * rpn_weight),
#                 loss_bbox=dict(
#                     type='IoULoss', linear=True,
#                     loss_weight=10.0 * rpn_weight))
#         ],
#         in_channels=256,
#         feat_channels=256,
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             scales=[8],
#             # ratios=[0.5, 1.0, 2.0],
#             ratios=[0.2, 0.5, 1.0, 2.0, 5.0],
#             strides=[4, 8, 16, 32, 64]),
#         bbox_coder=dict(
#             type='DeltaXYWHBBoxCoder',
#             target_means=[.0, .0, .0, .0],
#             target_stds=[1.0, 1.0, 1.0, 1.0]),
#         loss_cls=dict(
#             # type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#             type='FocalLoss', use_sigmoid=True, loss_weight=1.0), # 修改了loss，为了调控难易样本与正负样本比例
#         loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
#     roi_head=dict(
#         type='StandardRoIHead',
#         bbox_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         bbox_head=[
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 # num_classes=80,
#                 num_classes=6,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.1, 0.1, 0.2, 0.2]),
#                 reg_class_agnostic=False,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#                 loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 # num_classes=80,
#                 num_classes=6,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.05, 0.05, 0.1, 0.1]),
#                 reg_class_agnostic=False,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#                 loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 # num_classes=80,
#                 num_classes=6,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.033, 0.033, 0.067, 0.067]),
#                 reg_class_agnostic=False,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#                 loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
#     ]),

#     # model training and testing settings
#     train_cfg=dict(
#         rpn=[
#             dict(
#                 assigner=dict(
#                     type='RegionAssigner', center_ratio=0.2, ignore_ratio=0.5),
#                 allowed_border=-1,
#                 pos_weight=-1,
#                 debug=False),
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.7,
#                     neg_iou_thr=0.7,
#                     min_pos_iou=0.3,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=256,
#                     pos_fraction=0.5,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=False),
#                 allowed_border=-1,
#                 pos_weight=-1,
#                 debug=False)
#         ],
#         rpn_proposal=dict(
#         nms_across_levels=False,
#         nms_pre=2000,
#         nms_post=2000,
#         max_num=2000,
#         nms_thr=0.7,
#         min_bbox_size=0),
#         rcnn=[
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.4, # 更换
#                     neg_iou_thr=0.4,
#                     min_pos_iou=0.4,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='OHEMSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 pos_weight=-1,
#                 debug=False),
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.5,
#                     neg_iou_thr=0.5,
#                     min_pos_iou=0.5,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='OHEMSampler', # 解决难易样本，也解决了正负样本比例问题。
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 pos_weight=-1,
#                 debug=False),
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.6,
#                     neg_iou_thr=0.6,
#                     min_pos_iou=0.6,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='OHEMSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 pos_weight=-1,
#                 debug=False)
#         ],
#         stage_loss_weights=[1, 0.5, 0.25]),

#     test_cfg=dict(
#         rpn=dict(
#             nms_across_levels=False,
#             nms_pre=1000,
#             nms_post=1000,
#             max_num=1000,
#             nms_thr=0.7,
#             min_bbox_size=0),
#         rcnn=dict(
#             score_thr=0.05,
#             # nms=dict(type='nms', iou_threshold=0.5),
#             nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05),
#             max_per_img=100)
#         # soft-nms is also supported for rcnn testing
#         # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
#     ))

# optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))




# model settings
model = dict(
    type='CascadeRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        #dcn=dict( #在最后三个block加入可变形卷积 
         #   modulated=False, deformable_groups=1, fallback_on_stride=False),
          #  stage_with_dcn=(False, True, True, True)
        ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            # ratios=[0.5, 1.0, 2.0],
            ratios=[0.2, 0.5, 1.0, 2.0, 5.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, loss_weight=1.0), # 修改了loss，为了调控难易样本与正负样本比例
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                # num_classes=80,
                num_classes=6,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                # num_classes=80,
                num_classes=6,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                # num_classes=80,
                num_classes=6,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    ]))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler', 
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.4, # 更换
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                ignore_iof_thr=-1),
            sampler=dict(
                type='OHEMSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                type='OHEMSampler', # 解决难易样本，也解决了正负样本比例问题。
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                ignore_iof_thr=-1),
            sampler=dict(
                type='OHEMSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)
    ],
    stage_loss_weights=[1, 0.5, 0.25])
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=20)) # 这里可以换为soft_nms
# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../data/chongqing1_round1_train1_20191223/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(492,658), keep_ratio=True), #这里可以更换多尺度[(),()]
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
        img_scale=(492,658),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=8, # 有的同学不知道batchsize在哪修改，其实就是修改这里，每个gpu同时处理的images数目。
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'fixed_annotations.json', # 更换自己的json文件
        img_prefix=data_root + 'images/', # images目录
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'fixed_annotations.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'fixed_annotations.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001) # lr = 0.00125*batch_size，不能过大，否则梯度爆炸。
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[6, 12, 19])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=64,
    hooks=[
        dict(type='TextLoggerHook'), # 控制台输出信息的风格
        # dict(type='TensorboardLoggerHook') # 需要安装tensorflow and tensorboard才可以使用
    ])

