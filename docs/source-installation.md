# Build from Source

## Before building

Initialize submodules before building:

```shell
git submodule update --init --recursive
```

## Build with Docker

Use the provided Docker build helper:

```shell
CMAKE_BUILD_PARALLEL_LEVEL=4 ./scripts/docker_build_wheels.sh 311
```

SAPIEN currently targets PhysX `107.3-physx-5.6.1`. GPU-enabled builds require a CUDA toolkit and driver stack compatible with CUDA `>= 12.8`.

To build against locally extracted PhysX SDK archives, point the build at both the CPU and GPU packages before invoking the Docker helper:

```shell
export SAPIEN_PHYSX5_DIR=/path/to/physxcpu-linux-clang
export SAPIEN_PHYSX5_GPU_DIR=/path/to/physxgpu-linux-clang
CMAKE_BUILD_PARALLEL_LEVEL=4 ./scripts/docker_build_wheels.sh 311
```

The reference Dockerfile is available at [docker/Dockerfile](../docker/Dockerfile).

## Build without Docker

Building outside the provided Docker environment requires installing the same dependencies used by [docker/Dockerfile](../docker/Dockerfile). Once dependencies are available, build a wheel with:

```shell
export CUDA_PATH=/usr/local/cuda-12.8
export SAPIEN_PHYSX5_DIR=/path/to/physxcpu-linux-clang
export SAPIEN_PHYSX5_GPU_DIR=/path/to/physxgpu-linux-clang
python setup.py bdist_wheel --build-dir=sapien_build
```

## Validation

Use the narrowest test suite that matches your change. For PhysX-focused changes, run:

```shell
cd unittest
python -m unittest discover -s test_physx -p 'test_*.py'
```
