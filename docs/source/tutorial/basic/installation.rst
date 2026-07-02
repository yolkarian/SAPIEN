.. _installation:

Installation
==================

.. highlight:: python

SAPIEN is distributed as the ``sapien`` Python package on
`PyPI <https://pypi.org/project/sapien/>`_.

The checked-in build automation currently builds Linux wheels for Python 3.9,
3.10, 3.11, 3.12, and 3.13, and Windows wheels for Python 3.9, 3.10, 3.11,
and 3.12. Prefer these versions when installing released or nightly wheels.

Supported platforms and hardware depend on which SAPIEN features you use:

* Physics-only CPU simulation: Linux or Windows wheel matching your Python ABI.
* Rasterized rendering: a Vulkan-capable NVIDIA, AMD, or Intel GPU.
* PhysX GPU simulation: NVIDIA GPU with a driver/runtime stack compatible with
  CUDA ``>= 12.8``. Source builds with GPU support also need a matching CUDA
  toolkit.
* Ray tracing: a GPU and driver with Vulkan ray-tracing support.
* Denoising: ``oidn`` is available through the packaged renderer; ``optix``
  requires a compatible NVIDIA RTX stack.

Install from PyPI
-----------------------

.. code-block:: shell

   python -m pip install -U pip
   python -m pip install sapien

Verify installation
-----------------------

Server or headless machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The packaged offscreen example renders a red cube and writes
``sapien_offscreen.png`` to the current working directory.

.. code-block:: shell

   python -m sapien.example.offscreen

On a server without a display, Vulkan/EGL may print display-related warnings.
They can be ignored if the script exits successfully and the image is produced.

.. figure:: assets/example.offscreen.png
   :width: 120px
   :align: center
   :figclass: align-center

Desktop with display
^^^^^^^^^^^^^^^^^^^^^^^

The packaged viewer example opens a window showing a red cube on the ground.

.. code-block:: shell

   python -m sapien.example.hello_world

You can learn more about this scene in :ref:`hello_world`.

Build from source
-----------------------

Initialize submodules before any source build:

.. code-block:: shell

   git submodule update --init --recursive

Build in Docker
^^^^^^^^^^^^^^^^^^^^^^

The CI and helper scripts use ``yolkarian/sapien-build-env:0.3.1``. This is the
recommended Linux build environment.

.. code-block:: shell

   CMAKE_BUILD_PARALLEL_LEVEL=4 ./scripts/docker_build_wheels.sh 311

If the image is not available locally, pull it first:

.. code-block:: shell

   docker pull yolkarian/sapien-build-env:0.3.1

``scripts/docker_build_wheels.sh`` forwards ``SAPIEN_PHYSX5_DIR``,
``SAPIEN_PHYSX5_GPU_DIR``, and ``SAPIEN_PHYSX5_VERSION`` into the container. Use
these variables when verifying against local PhysX SDK archives instead of the
CMake download path.

.. code-block:: shell

   export SAPIEN_PHYSX5_DIR=/path/to/physxcpu-linux-clang
   export SAPIEN_PHYSX5_GPU_DIR=/path/to/physxgpu-linux-clang
   CMAKE_BUILD_PARALLEL_LEVEL=4 ./scripts/docker_build_wheels.sh 311

Build without Docker
^^^^^^^^^^^^^^^^^^^^^^

Native builds should use the same PhysX and CUDA toolchain versions as the
Docker image. SAPIEN currently targets PhysX ``107.3-physx-5.6.1``.

.. code-block:: shell

   export CUDA_PATH=/usr/local/cuda-12.8
   export SAPIEN_PHYSX5_DIR=/path/to/physxcpu-linux-clang
   export SAPIEN_PHYSX5_GPU_DIR=/path/to/physxgpu-linux-clang
   python setup.py bdist_wheel --build-dir=sapien_build

For a direct CI-style build, use ``scripts/build.sh`` and limit parallelism on
shared machines:

.. code-block:: shell

   ./scripts/build.sh 311 --jobs 4

Validation
-----------------------

After building a wheel, install it in a clean Python environment and run the
narrowest matching tests. For PhysX-focused changes, the repository test suite
is:

.. code-block:: shell

   cd unittest
   python -m unittest discover -s test_physx -p 'test_*.py'

The documentation API pages are generated from an installed ``sapien`` package,
so ``make -C docs html`` assumes the package imports successfully and that the
Sphinx tools are installed.
