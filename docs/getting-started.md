# Getting Started

Install SAPIEN from a wheel published on [GitHub Releases](https://github.com/yolkarian/SAPIEN/releases). Download the wheel that matches your Python version and platform, then install it with pip:

```shell
python -m pip install -U pip
python -m pip install ./sapien-*.whl
```

SAPIEN requires Linux with an NVIDIA, AMD, or Intel GPU for rendering. Verify the installation with:

```shell
python -m sapien.example.hello_world
```

Then follow the tutorials at [https://sapien-sim.github.io/docs/](https://sapien-sim.github.io/docs/).

Useful local documentation entry points:

- [Sphinx documentation source](source/index.md)
- [Basic tutorial](source/tutorial/basic/index.md)
- [Rendering tutorial](source/tutorial/rendering/index.md)
- [Robotics tutorial](source/tutorial/robotics/index.md)
- [Reinforcement learning tutorial](source/tutorial/rl/index.md)
