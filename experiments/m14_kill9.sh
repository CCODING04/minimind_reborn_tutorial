#!/bin/bash
# 从零启动 tiny 训练，step>30 后对整个进程组 kill -9（uv 包装与 python 子进程一锅端）
cd /home/admin02/Code/WorkSpace/PythonFormat/Reproduce/minimind_reborn
rm -f checkpoints/pretrain_64_resume.pth
ls -d runs/smoke_pretrain/*/ > /tmp/m14_before.txt
setsid uv run python tools/train.py configs/smoke/smoke_pretrain.yaml \
  --set train.max_steps=60 --set train.save_interval_steps=10 \
  --set train.log_interval=2 --set train.resume=false > /tmp/m14_B_killed.log 2>&1 &
TRAIN_PID=$!
LOG=""
for i in $(seq 1 300); do
  NEW=$(ls -d runs/smoke_pretrain/*/ 2>/dev/null | grep -v -F -f /tmp/m14_before.txt | head -1)
  if [ -n "$NEW" ] && [ -f "$NEW/train.log" ]; then LOG="$NEW/train.log"; break; fi
  kill -0 $TRAIN_PID 2>/dev/null || break
  sleep 0.02
done
echo "watching: $LOG"
for i in $(seq 1 500); do
  if grep -q "(34/" "$LOG" 2>/dev/null; then
    PGID=$(ps -o pgid= -p $TRAIN_PID | tr -d ' ')
    kill -9 -$PGID     # 杀整个进程组：uv 包装 + python 子进程
    echo "kill -9 sent to process group $PGID (step 34 logged; step-30 ckpt on disk)"
    break
  fi
  kill -0 $TRAIN_PID 2>/dev/null || { echo "process exited before kill"; break; }
  sleep 0.02
done
wait $TRAIN_PID 2>/dev/null
echo "shell exit code after kill: $?"
sleep 1
echo "---- train.log tail（应止于 ~step 34，无训练结束行）----"
tail -3 "$LOG"
echo "---- 进程残活检查 ----"
pgrep -f "tools/train.py" | wc -l
