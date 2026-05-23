#!/bin/bash
# Jetson Nano 部署脚本

set -e

echo "======================================"
echo "Jetson Nano 异常检测 Docker 部署"
echo "======================================"

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo "错误: Docker 未安装"
    exit 1
fi

# 检查NVIDIA Docker支持
if ! command -v nvidia-docker &> /dev/null; then
    echo "警告: nvidia-docker 未安装，尝试使用 docker"
    DOCKER_CMD="docker"
else
    DOCKER_CMD="nvidia-docker"
fi

# 选择要使用的模型
MODEL_DIR="./results/models"
echo "可选的模型:"
ls -1 "$MODEL_DIR" | head -20

read -p "输入要使用的模型路径 (默认: ./results/models/10_gpu0_repeat1/model_s.pth): " MODEL_PATH
MODEL_PATH=${MODEL_PATH:-"./results/models/10_gpu0_repeat1/model_s.pth"}

# 构建Docker镜像
echo "正在构建Docker镜像..."
docker build -t jetson-anomaly-detection:latest .

# 运行容器
echo "正在启动容器..."
docker run --rm --network host \
    --runtime nvidia \
    -e DISPLAY=$DISPLAY \
    -v $(pwd)/results:/workspace/results \
    -v $(pwd)/realtime_inference.py:/workspace/realtime_inference.py \
    jetson-anomaly-detection:latest \
    python3 /workspace/realtime_inference.py --model_path "$MODEL_PATH"

echo "部署完成!"
