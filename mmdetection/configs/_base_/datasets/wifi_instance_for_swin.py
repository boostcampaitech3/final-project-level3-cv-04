# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/final-project-level3-cv-04/Data set/'
# CLASSES = ("ID", "PW")
# PALETTE = [(220, 20, 60), (119, 11, 32)] #, (0, 0, 142)] # , (0, 0, 230)]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_general.json',
        # ann_file=data_root + 'general_00_213.json',
        # ann_file=data_root + 'annotations/general_11_30.json',
        img_prefix=data_root + 'real data/general/', 
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'valid_general.json',
        img_prefix=data_root + 'real data/general/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'valid_general.json',
        img_prefix=data_root + 'real data/general/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
