_base_ = '../faster_rcnn/faster_rcnn_r50_1x_coco_origin_noclahe.py'
rpn_weight = 0.7
model = dict(
    #pretrained='torchvision://resnet50',
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(3, ),
        strides=(1, 2, 2, 1),
        dilations=(1, 1, 1, 2),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    neck=None,
    rpn_head=dict(
        type='RPNHead',
        in_channels=2048,
        feat_channels=2048,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[1.0],
            strides=[16]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        
        roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=2048,
            featmap_strides=[16]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=2048,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0))
    ),
    # model training and testing settings
    train_cfg=dict(
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
                    add_gt_as_proposals=False),  # 把ground truth加入proposal作为正样本
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
        rpn_proposal=dict(max_num=500, nms_thr=0.5),
        rcnn=dict(
            assigner=dict(
                pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5),
            sampler=dict(type='RandomSampler', num=256))),
    test_cfg=dict(
        rpn=dict(nms_across_levels=False,max_num=300, nms_thr=0.3), 
        rcnn=dict(score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0),
            max_per_img=100)
        ))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
