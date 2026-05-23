#!/bin/bash

# 并行训练脚本 - 在Linux上运行
# 4个GPU并行，每个GPU一次只跑一个pattern

echo "========================================"
echo "  启动并行训练任务"
echo "========================================"
echo ""

PATTERNS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

run_gpu_task() {
    local gpu_id=$1
    local repeat=$2
    for pattern in "${PATTERNS[@]}"; do
        echo "GPU $gpu_id: Starting pattern $pattern (repeat $repeat)"
        python b_run.py --phase train --num_of_pattern $pattern --gpu_id $gpu_id --repeat $repeat
        echo "GPU $gpu_id: Finished pattern $pattern (repeat $repeat)"
    done
}

run_gpu_task 0 1 &
PID1=$!
run_gpu_task 1 2 &
PID2=$!
run_gpu_task 2 3 &
PID3=$!
run_gpu_task 3 4 &
PID4=$!

echo "已启动训练任务:"
echo "  GPU 0: patterns 1-19, repeat 1"
echo "  GPU 1: patterns 1-19, repeat 2"
echo "  GPU 2: patterns 1-19, repeat 3"
echo "  GPU 3: patterns 1-19, repeat 4"
echo ""
echo "等待所有任务完成..."
echo ""

wait $PID1
wait $PID2
wait $PID3
wait $PID4

echo ""
echo "========================================"
echo "  所有训练任务已完成!"
echo "========================================"