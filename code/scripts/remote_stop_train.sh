#!/bin/bash
# 立即停止循环训练与当前正在跑的 epoch
echo "停止循环守卫 ..."
pkill -f "remote_run_forever.sh" 2>/dev/null
pkill -f "run_forever.sh" 2>/dev/null
touch /root/autodl-tmp/dmm/STOP
sleep 1
echo "停止训练进程 ..."
pkill -f "tools/train.py" 2>/dev/null
sleep 3
echo "剩余进程："
ps aux | grep -E "[t]ools/train.py|[r]un_forever" | head -n 5
echo "已停止。已保存的权重仍在 work_dirs/ 下。"
