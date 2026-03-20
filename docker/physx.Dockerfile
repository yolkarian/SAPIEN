FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN mkdir /workspace
RUN apt update && apt install -y git cmake curl wget libstdc++6 clang-10 g++-9 libx11-dev
RUN cd /workspace && wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run && \
    sh cuda_12.8.0_570.86.10_linux.run --silent --toolkit --toolkitpath=/workspace/cuda --override && rm -f cuda_12.8.0_570.86.10_linux.run
ENV CUDA_PATH=/workspace/cuda CUDACXX=/workspace/cuda/bin/nvcc PATH="/workspace/cuda/bin:$PATH" LD_LIBRARY_PATH="/workspace/cuda/lib64:$LD_LIBRARY_PATH"

WORKDIR /workspace
RUN git clone https://github.com/NVIDIA-Omniverse/PhysX.git
