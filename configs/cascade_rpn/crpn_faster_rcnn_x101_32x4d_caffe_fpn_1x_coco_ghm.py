_base_ = '../faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco.py'
# 

rpn_weight = 0.7
model = dict(
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        end_level = 3,
        num_outs=3),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4],
            ratios=[1.0],
            strides=[4, 8, 16]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0 * rpn_weight),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=10 * rpn_weight)),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(target_stds=[0.04, 0.04, 0.08, 0.08]),
            loss_cls=dict(
                type='GHMC',
                bins=30,
                momentum=0.75,
                use_sigmoid=True,
                loss_weight=1.0),
            loss_bbox=dict(
                type='GHMR',
                mu=0.02,
                bins=10,
                momentum=0.7,
                loss_weight=10.0)
            )),
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
        rpn=dict(nms_across_levels=False,max_num=300, nms_thr=0.3,min_bbox_size=1), 
        rcnn=dict(score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0),
            max_per_img=100)
        ))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
