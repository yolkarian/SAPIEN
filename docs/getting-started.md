# Getting Started

Install SAPIEN from a wheel published on this fork's
[GitHub Releases](https://github.com/yolkarian/SAPIEN/releases). Download the
wheel matching your Python ABI and platform, then install it with pip:

```shell
python -m pip install -U pip
python -m pip install ./sapien-*.whl
```

Verify a headless installation with the offscreen renderer:

```shell
python -m sapien.example.offscreen
```

It should create `sapien_offscreen.png`. On a desktop, verify the interactive
Viewer with:

```shell
python -m sapien.example.hello_world
```

Continue with the current documentation in this repository:

- [Installation and platform requirements](source/tutorial/basic/installation.md)
- [Basic tutorial](source/tutorial/basic/index.md)
- [Rendering tutorial](source/tutorial/rendering/index.md)
- [Robotics tutorial](source/tutorial/robotics/index.md)
- [Reinforcement-learning tutorial](source/tutorial/rl/index.md)
- [Motion-planning tutorial](source/tutorial/motion_planning/index.md)
- [SAPIEN 1.x/2.x migration notes](source/tutorial/migration/index.md)

The upstream published site can describe a different SAPIEN release. For this
fork, prefer the checked-in docs and the GitHub Pages site linked from
[`readme.md`](../readme.md).
