#!/bin/bash
# 循环训练：每 12 个 epoch 训练 + 验证为一轮，跑完自动开始下一轮，
# 直到手动停止（touch STOP 文件，或执行 stop_train.sh）。
#
# 用法：./run_forever.sh [输出根目录]
#   默认输出根目录 work_dirs/loop_pt
#
# 停止方式：
#   1) 优雅停止（跑完当前轮再停）：  touch /root/autodl-tmp/dmm/STOP
#   2) 立即停止（连当前轮一起杀）：  bash /root/autodl-tmp/dmm/stop_train.sh

cd /root/autodl-tmp/dmm || exit 1

P=/root/miniconda3/bin/python
export PYTHONPATH=/root/autodl-tmp/dmm/code:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export DMM_VEDAI_ROOT=/root/autodl-tmp/dmm/datasets/vedai
# VMamba ImageNet 预训练权重（首轮据此初始化主干）
export DMM_VMAMBA_CKPT=/root/autodl-tmp/dmm/weights/vssm/vssm_tiny_0230_ckpt_epoch_262.pth

CONFIG=code/configs/dmm/dmm_s2anet_1x_vedai.py
BASE=${1:-work_dirs/loop_pt}
STOP=/root/autodl-tmp/dmm/STOP
MAX_FAIL=3

mkdir -p "$BASE"
rm -f "$STOP"

echo "[$(date '+%F %T')] 输出目录：$BASE"
echo "[$(date '+%F %T')] 预训练权重：$DMM_VMAMBA_CKPT"
if [ ! -f "$DMM_VMAMBA_CKPT" ]; then
    echo "[$(date '+%F %T')] 警告：预训练权重不存在，主干将从头训练"
fi

echo "[$(date '+%F %T')] 等待已存在的训练进程结束 ..."
while pgrep -f "tools/train.py" > /dev/null 2>&1; do
    sleep 20
done

CYCLE=1
INIT=""
FAIL=0
while [ ! -f "$STOP" ]; do
    WD="$BASE/cycle_$CYCLE"
    LOG="$BASE/cycle_$CYCLE.log"
    mkdir -p "$WD"

    if [ -n "$INIT" ]; then
        SRC="从 $INIT 继续"
    else
        SRC="从头训练（主干加载 ImageNet 预训练）"
    fi

    echo "[$(date '+%F %T')] ===== 第 $CYCLE 轮开始（$SRC，12 epoch）=====" | tee -a "$BASE/loop.log"

    if [ -n "$INIT" ]; then
        $P code/tools/train.py "$CONFIG" --work-dir "$WD" \
            --cfg-options "load_from=$INIT" default_hooks.checkpoint.max_keep_ckpts=2 > "$LOG" 2>&1
    else
        $P code/tools/train.py "$CONFIG" --work-dir "$WD" \
            --cfg-options default_hooks.checkpoint.max_keep_ckpts=2 > "$LOG" 2>&1
    fi
    CODE=$?

    if [ $CODE -ne 0 ]; then
        FAIL=$((FAIL + 1))
        echo "[$(date '+%F %T')] 第 $CYCLE 轮失败（退出码 $CODE），连续失败 $FAIL 次，见 $LOG" | tee -a "$BASE/loop.log"
        if [ $FAIL -ge $MAX_FAIL ]; then
            echo "[$(date '+%F %T')] 连续失败 $MAX_FAIL 次，停止循环。" | tee -a "$BASE/loop.log"
            exit 1
        fi
        sleep 60
        continue
    fi
    FAIL=0

    NEXT=""
    for cand in "$WD/epoch_12.pth" "$WD"/best_dota_mAP_epoch_*.pth; do
        if [ -f "$cand" ]; then NEXT="$cand"; break; fi
    done
    if [ -z "$NEXT" ]; then
        echo "[$(date '+%F %T')] 第 $CYCLE 轮没找到权重文件，停止。" | tee -a "$BASE/loop.log"
        exit 1
    fi

    MAP=$(grep -E "Epoch\(val\)" "$LOG" | tail -n 1 | sed -n 's/.*dota\/mAP: \([0-9.]*\).*/\1/p')
    LOADED=$(grep -c "Successfully load ckpt" "$LOG")
    echo "[$(date '+%F %T')] 第 $CYCLE 轮完成，验证 mAP=$MAP，权重 $NEXT（预训练加载次数 $LOADED）" | tee -a "$BASE/loop.log"

    INIT="$NEXT"
    CYCLE=$((CYCLE + 1))
    sleep 5
done

echo "[$(date '+%F %T')] 检测到 STOP 文件，循环结束。" | tee -a "$BASE/loop.log"
