#!/bin/bash

# ================= 配置区 =================
TOTAL_JOBS=50            # 你想要分成的总份数 (根据你的CPU核心数调整)
SCRIPT_NAME="simulate_L5PC_and_create_dataset.py"
LOG_DIR="logs_pairs"     # 日志存放目录
# =========================================

# 1. 准备工作
mkdir -p $LOG_DIR
echo "Starting $TOTAL_JOBS parallel jobs..."
echo "Logs will be saved to: $LOG_DIR/"

# 2. 循环启动任务
for (( i=0; i<TOTAL_JOBS; i++ ))
do
    echo "Launching Job ID: $i / $((TOTAL_JOBS-1))"

    # nohup: 防止断开SSH后任务终止
    # &: 在后台运行
    # > ... 2>&1: 将标准输出和错误输出都保存到日志
    nohup python $SCRIPT_NAME \
        --mode pairs \
        --job_id $i \
        --total_jobs $TOTAL_JOBS \
        > "$LOG_DIR/job_${i}.log" 2>&1 &

    # 可选：每启动一个暂停 0.1秒，防止瞬间 CPU 冲击
    sleep 0.1
done

echo "--------------------------------------------------"
echo "All jobs launched! Use 'top' or 'htop' to monitor."
echo "To kill all jobs if needed: pkill -f $SCRIPT_NAME"