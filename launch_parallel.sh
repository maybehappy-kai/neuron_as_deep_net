#!/bin/bash

# ================= 配置区 =================
TOTAL_JOBS=50
SMOKE_TEST=1             # 设置为 1 开启冒烟测试，设置为 0 进行全量实验
SCRIPT_NAME="simulate_L5PC_and_create_dataset.py"
MERGE_SCRIPT="merge_fast.py"
# =========================================

# 1. 动态生成带时间戳的输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ "$SMOKE_TEST" -eq 1 ]; then
    OUT_DIR="results_${TIMESTAMP}_smoke"
    SMOKE_FLAG="--smoke_test"
    echo "========================================="
    echo " 🚀 RUNNING IN SMOKE TEST MODE"
    echo "========================================="
else
    OUT_DIR="results_${TIMESTAMP}_full"
    SMOKE_FLAG=""
    echo "========================================="
    echo " 💥 RUNNING IN FULL EXPERIMENT MODE"
    echo "========================================="
fi

LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
echo "All data and logs will be saved to: $OUT_DIR/"

# 2. 先运行 Setup 模式 (获取 baseline 和 singles)
echo "Running SETUP phase..."
python $SCRIPT_NAME --mode setup --out_dir "$OUT_DIR" $SMOKE_FLAG > "$LOG_DIR/setup.log" 2>&1

# 3. 循环启动 Pairs 并行任务
echo "Launching $TOTAL_JOBS parallel jobs for PAIRS phase..."
for (( i=0; i<TOTAL_JOBS; i++ ))
do
    nohup python $SCRIPT_NAME \
        --mode pairs \
        --job_id $i \
        --total_jobs $TOTAL_JOBS \
        --out_dir "$OUT_DIR" \
        $SMOKE_FLAG \
        > "$LOG_DIR/job_${i}.log" 2>&1 &

    sleep 0.1
done

echo "--------------------------------------------------"
echo "All jobs launched in background."
echo "You can monitor the progress with: tail -f $LOG_DIR/job_0.log"
echo "Once all jobs complete, merge them by running:"
echo "python $MERGE_SCRIPT --data_dir $OUT_DIR"