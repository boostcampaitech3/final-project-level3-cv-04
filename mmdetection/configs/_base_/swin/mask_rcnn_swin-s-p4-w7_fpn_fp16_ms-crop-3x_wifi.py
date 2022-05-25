_base_ = './mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)))

classes = ("ID", "PW")
CLASSES = ("ID", "PW")

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