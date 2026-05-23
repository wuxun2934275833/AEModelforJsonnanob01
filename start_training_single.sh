#!/bin/bash

# 并行训练脚本 - 4块GPU轮询训练15个pattern，每个pattern只训练一次
# 训练完成后自动生成热力图

echo "========================================"
echo "  启动4 GPU并行训练任务"
echo "  每个Pattern训练1次，共15个Pattern"
echo "========================================"
echo ""

PATTERNS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
NUM_GPUS=4
GPU_IDS=(0 1 2 3)

run_pattern() {
    local gpu_id=$1
    local pattern=$2
    echo "GPU $gpu_id: Starting pattern $pattern"
    python b_run.py --phase train --num_of_pattern $pattern --num_of_epochs 150 --early_stop_patience 15 --gpu_id $gpu_id --repeat 1
    echo "GPU $gpu_id: Finished pattern $pattern"
}

export -f run_pattern

echo "计算任务分配..."
echo ""

# 构建任务列表
TASKS=()
for pattern in "${PATTERNS[@]}"; do
    gpu_id=${GPU_IDS[$(($pattern % NUM_GPUS))]}
    TASKS+=("$gpu_id $pattern")
done

# 使用GNU parallel或简单循环进行并行任务分配
# 方法1: 使用后台进程并行
PIDS=()
for i in "${!PATTERNS[@]}"; do
    pattern=${PATTERNS[$i]}
    gpu_id=${GPU_IDS[$(($pattern % NUM_GPUS))]}

    echo "分配: Pattern $pattern -> GPU $gpu_id"

    run_pattern $gpu_id $pattern &
    PIDS+=($!)
done

echo ""
echo "已启动所有训练任务"
echo "共 ${#PATTERNS[@]} 个Pattern，分布在 $NUM_GPUS 块GPU上"
echo ""
echo "等待所有任务完成..."
echo ""

# 等待所有后台任务完成
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo ""
echo "========================================"
echo "  所有训练任务已完成!"
echo "  热力图已保存到 ./images 目录"
echo "========================================"
