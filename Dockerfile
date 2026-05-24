# Jeton Nano Docker 镜像
# 使用 NVIDIA L4T PyTorch 镜像作为基础

FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3

# 设置工作目录
WORKDIR /workspace

# 修复apt源签名问题 - 注释掉kitware源
RUN sed -i 's/^deb.*kitware.*/# disabled kitware source/' /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null || true

# 安装必要的系统包和Python依赖
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libglib2.0-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    python3-pip \
    libyaml-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir \
    scipy \
    pillow \
    pyyaml

# 复制推理所需的项目文件（只保留部署必需的内容）
COPY b_models_and_sub_models /workspace/b_models_and_sub_models
COPY c_utils /workspace/c_utils
COPY results /workspace/results
COPY realtime_inference.py /workspace/realtime_inference.py
COPY config.yaml /workspace/config.yaml

# 设置环境变量
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV DISPLAY=:0

# 暴露端口
EXPOSE 8888

# 启动命令
CMD ["python3", "/workspace/realtime_inference.py", "--camera", "0"]
