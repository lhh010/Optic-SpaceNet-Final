# Gazelle 决赛演示环境 (CPU torch + fastapi + ssh/串口直连工具链)
# 构建: docker build -t gazelle-demo:1.0 -f tools/Dockerfile .
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        sshpass openssh-client ca-certificates picocom usbutils && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy pillow fastapi uvicorn httpx pydantic pyserial
RUN pip install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torchvision \
        --index-url https://download.pytorch.org/whl/cpu

WORKDIR /workspace
CMD ["/bin/bash"]
