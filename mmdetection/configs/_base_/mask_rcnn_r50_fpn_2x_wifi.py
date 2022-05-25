_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/wifi_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

classes = ("ID", "PW")
CLASSES = ("ID", "PW")
# PALETTE = [(220, 20, 60), (119, 11, 32)] #, (0, 0, 142)] # , (0, 0, 230)]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        # type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes),
        # ann_file='path/to/your/train/annotation_data',
        # img_prefix='path/to/your/train/image_data'),
    val=dict(
        # type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes),
        # ann_file='path/to/your/val/annotation_data',
        # img_prefix='path/to/your/val/image_data'),
    test=dict(
        # type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes))
        # ann_file='path/to/your/test/annotation_data',
        # img_prefix='path/to/your/test/image_data'))
        
model = dict(
    roi_head=dict(
        bbox_head=
            dict(
                type='Shared2FCBBoxHead',
                num_classes=2),
        mask_head=dict(
            type='FCNMaskHead',
            num_classes=2)
        ))

optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))