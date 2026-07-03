# Server Rendering

## Offscreen rendering on a server

For GPU servers without a display, install the system packages `libegl1` and `libxext6`. In NVIDIA Docker environments, enable graphics, utility, and compute capabilities:

```Dockerfile
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
```

Verify SAPIEN with:

```shell
python -m sapien.example.hello_world
```

## Virtual desktop on a server

For GPU servers with a virtual desktop, additionally install `xvfb`, `x11vnc`, and a window manager such as `fluxbox` or `xfce`. Enable display capabilities in NVIDIA Docker environments:

```Dockerfile
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute,display
```

For example, with `fluxbox`, start a VNC server with:

```shell
x11vnc -create -env FD_PROG=/usr/bin/fluxbox -env X11VNC_FINDDISPLAY_ALWAYS_FAILS=1 -env X11VNC_CREATE_GEOM=${99:-1920x1080x16} -gone 'pkill Xvfb' -nopw
# Note: use a strong password and/or only allow local access.
```

Connect to port 5900, then verify SAPIEN with:

```shell
python -m sapien.example.hello_world
```
