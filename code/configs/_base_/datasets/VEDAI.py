# VEDAI 可见光 / 红外图像对的数据配置
# 图像与标签由 tools/data_process.py 和 tools/VOC2DOTA.py 生成
import os

# 相对于工作目录；train_vedai.py 会自动切到仓库根目录。
# 若数据放在别处，可设置环境变量 DMM_VEDAI_ROOT 覆盖
dataset_type = "mmrotate.VEDAIDataset"
data_root = os.environ.get("DMM_VEDAI_ROOT", "datasets/vedai")
backend_args = None

train_pipeline = [
    dict(type="mmrotate.LoadRGBTImageFromFile", backend_args=backend_args),
    dict(type="mmrotate.LoadRGBTAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox", gt_bboxes2="rbox")),
    dict(type="mmrotate.ResizeRGBT", scale=(640, 512), keep_ratio=True),
    dict(
        type="mmrotate.RandomFlipRGBT",
        prob=0.5,
        direction=["horizontal", "vertical", "diagonal"],
    ),
    dict(type="mmdet.FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type="mmrotate.PackRGBTDetInputs",
        meta_keys=("img_id", "img_path", "img_path2", "ori_shape", "img_shape", "scale_factor"),
    ),
]
val_pipeline = [
    dict(type="mmrotate.LoadRGBTImageFromFile", backend_args=backend_args),
    dict(type="mmrotate.LoadRGBTAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox", gt_bboxes2="rbox")),
    dict(type="mmrotate.ResizeRGBT", scale=(640, 512), keep_ratio=True),
    dict(
        type="mmrotate.PackRGBTDetInputs",
        meta_keys=("img_id", "img_path", "img_path2", "ori_shape", "img_shape", "scale_factor"),
    ),
]
test_pipeline = [
    dict(type="mmrotate.LoadRGBTImageFromFile", backend_args=backend_args),
    dict(type="mmrotate.ResizeRGBT", scale=(640, 512), keep_ratio=True),
    dict(
        type="mmrotate.PackRGBTDetInputs",
        meta_keys=("img_id", "img_path", "img_path2", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    # 8 GB 显存下 batch=2 会溢出到共享显存，速度极慢；这里固定为 1
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="train/annofiles_ir",
        aux_ann_file="train/annofiles_vi",
        data_prefix=dict(
            img_path="train/image_vi",
            img_path2="train/image_ir",
        ),
        img_suffix="png",
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="val/annofiles_ir",
        aux_ann_file="val/annofiles_vi",
        data_prefix=dict(
            img_path="val/image_vi",
            img_path2="val/image_ir",
        ),
        img_suffix="png",
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="test/annofiles_ir",
        aux_ann_file="test/annofiles_vi",
        data_prefix=dict(
            img_path="test/image_vi",
            img_path2="test/image_ir",
        ),
        img_suffix="png",
        test_mode=True,
        pipeline=val_pipeline,
    ),
)

val_evaluator = dict(type="DOTAMetric", metric="mAP")
test_evaluator = val_evaluator
