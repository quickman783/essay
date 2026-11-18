# DMM + DCFL + MBO

放入现有 `quickman783/essay` 仓库的 MBO 扩展。复用 `code/mmrotate/` 和 `code/tools/`，不包含第二套 DMM 工程。模型、损失、标签分配和学习率策略沿用已完成的 MBO 实验。

## 放在哪里

```text
essay/
├── train_dcfl_mbo_dronevehicle_loop.py
├── code/
│   ├── mmrotate/                   原仓库的 DMM 基础代码
│   ├── tools/                      原仓库的底层 train.py、test.py
│   ├── dmm_dcfl/                   原仓库的 DCFL 扩展，保留
│   └── dmm_gaucho/                 原仓库的 GauCho 扩展，保留
└── add_new_code/
    └── dmm_mbo/                    本扩展
        ├── detector.py             三分支模型与损失组织
        ├── balancing.py            逐实例动态 λ
        ├── head_loss.py            实例代价与加权损失
        ├── dcfl_targets.py         DCFL 目标与 GT 身份
        ├── fast_assigner.py        等价加速分配器
        ├── dcfl/                   MBO 使用的 DCFL 适配，不修改原 dmm_dcfl
        ├── training/               自动训练、评估、进度、指标记录
        └── configs/                MBO 配置及必要的基类配置
```

`check_integration.py` 用于真实 batch 检查；`export_inference.py` 用于去掉辅助参数。`server_tools/` 单独存放可复用服务器工具，其中 `checkpoint_storage.py` 负责可选的分盘保存，默认不导入或启用。

训练配置只有 `configs/dmm_dcfl_mbo_train.py`，默认使用 Fast 分配器和 VMamba 预训练初始化；推理配置为 `configs/dmm_dcfl_mbo_inference.py`。继承依赖统一放在 `configs/_base_/`，需要一起上传。旧命令中的 `--fast-assign --pretrained` 仍兼容，日常不必再加。

## 安装与数据

使用原 DMM CUDA 环境。既有实验的核心版本为 Python 3.10、PyTorch 2.1.2+cu118、MMCV 2.1.0、MMEngine 0.10.5、MMDetection 3.3.0、NumPy 1.26.4。需要 VMamba 的 `selective_scan_cuda_oflex` 扩展；Python 补充依赖见本目录 `requirements.txt`。

本扩展复用已有 DMM/VMamba 环境，不补充或修改 VMamba 源码。

数据使用准备好的 RGB/IR 图像及 DOTA 格式标注：

```text
dronevehicle/{train,val,test}/{image_vi,image_ir,annofiles_vi,annofiles_ir}/
```

在仓库根目录设置：

```bash
export DMM_DV_ROOT=/absolute/path/to/dronevehicle
export DMM_TRAIN_SEED=1017387474
```

使用 `--pretrained` 时，原仓库布局中的预训练文件位置是：

```text
code/weights/vssm/vssm_tiny_0230_ckpt_epoch_262.pth
```

本地混合实验工作区尚保留 `code/DMM-main/` 布局时，入口会自动使用它的基础代码、工具和 `weights/`。上传原 `essay` 仓库时不需要上传 `DMM-main`，会直接使用已有 `code/mmrotate/`。

## 训练

在 `essay` 根目录执行：

```bash
# 只检查路径，不训练
python train_dcfl_mbo_dronevehicle_loop.py --max-cycles 1 --dry-run

# 训练 12 轮，保留每轮权重，并自动评估 val/test、记录各类 AP 和 CSV/XLSX
python train_dcfl_mbo_dronevehicle_loop.py \
  --max-cycles 1 --keep-epochs -1 --output-root /absolute/path/to/new_outputs
```

初始峰值学习率 `1e-4`，1000 步 warmup，epoch 8/11 后各降十倍。下一 cycle 峰值乘 0.5，即第二阶段 `5e-5`。AdamW、batch size=2、FP32 与原实验相同。

从完整 MBO 检查点开始第二个 cycle：

```bash
python train_dcfl_mbo_dronevehicle_loop.py \
  --init /absolute/path/to/epoch12.pth --cycle 2 --max-cycles 2 \
  --keep-epochs -1 --output-root /absolute/path/to/new_cycle2_outputs
```

`--max-cycles` 是绝对 cycle 上限。新起点使用新输出目录，避免已有 `run_state.json` 优先接续旧状态。`--init` 加载模型开始新 cycle，重新初始化优化器和调度器。

## 测试与整网检查

原仓库中执行：

```bash
export PYTHONPATH="$PWD/add_new_code:$PWD/code"

python add_new_code/dmm_mbo/training/evaluate.py \
  --config add_new_code/dmm_mbo/configs/dmm_dcfl_mbo_train.py \
  --checkpoint /absolute/path/to/mbo.pth --split test \
  --out work_dirs/mbo_eval/test_metrics.json --work-dir work_dirs/mbo_eval/runtime

python add_new_code/dmm_mbo/check_integration.py \
  --config add_new_code/dmm_mbo/configs/dmm_dcfl_mbo_train.py \
  --data-root "$DMM_DV_ROOT" --report work_dirs/mbo_checks/integration.json
```

普通 predict 只走 Fusion；导出去掉辅助参数：

```bash
python add_new_code/dmm_mbo/export_inference.py /path/to/mbo.pth /path/to/new_fusion_only.pth
```

## 配置与算法

`configs/_base_/dmm_dcfl_mbo_loop_dronevehicle.py` 设置 MBO：`temperature=1`、`aux_loss_weight=1`、`cost_class=1`、`cost_bbox=1`。

每个 GT 三路均有匹配时，`lambda = 3 * softmax(cost / temperature)`；更大代价获得更大监督权重。λ 不反传，作用于对应正样本的分类与五维旋转回归。三路用同一 IR GT 身份；保留原 DCFL、S2A 头、le135 编码和 Focal/SmoothL1，不引入 LCF 或 Hungarian matching。

参考 [DMM](https://github.com/Another-0/DMM)、[MS-DETR](https://github.com/YinghuiXing/MS-DETR)、MMRotate、VMamba 及既有 DCFL 适配。保留原仓库许可证与源码版权说明。

## 上传范围

只新增/更新下面两处：

```text
train_dcfl_mbo_dronevehicle_loop.py
add_new_code/dmm_mbo/
```

在原仓库根目录添加：

```bash
git add train_dcfl_mbo_dronevehicle_loop.py add_new_code/dmm_mbo
git diff --cached --stat
```

数据、训练权重、预训练文件、日志和缓存均不在上传包中。本次没有提交或推送 GitHub。
