(server_rendering)=

# Server Rendering

## Offscreen rendering without a display

SAPIEN renders offscreen through Vulkan; it does not require an X server. Install
the Vulkan/EGL loader packages required by your distribution (for example,
`libvulkan1`, `libegl1`, `libgl1`, and `libxext6` on Ubuntu).

In an NVIDIA container, expose graphics in addition to compute and utility:

```Dockerfile
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

NVIDIA Container Toolkit supplies the driver libraries. If the base image does
not contain ICD manifests, add the manifests, not a second copy of the driver.
Install this as `/etc/vulkan/icd.d/nvidia_icd.json`:

```json
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "libGLX_nvidia.so.0",
    "api_version": "1.1.95"
  }
}
```

For EGL, install `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` with
`"library_path": "libEGL_nvidia.so.0"`.

Verify offscreen rendering with:

```shell
python -m sapien.example.offscreen
```

The command must write `sapien_offscreen.png`. Do not ignore
`Failed to find Vulkan ICD file`: fix ICD discovery and ensure
`VK_ICD_FILENAMES` points to the NVIDIA manifest. Avoid Mesa software ICDs for
NVIDIA GPU validation.

## PhysX GPU in containers

The offscreen example tests rendering, not PhysX GPU. A first call to
`sapien.physx.enable_gpu()` downloads the matching PhysX GPU library to
`~/.sapien/physx/<version>` when it is absent. Fresh containers lose that
writable layer, so production images should extract the matching
`physxgpu-linux-clang.zip` into that directory during the image build, or mount
a persistent home volume. Do not call `enable_gpu()` while building the image;
it loads `libcuda.so` and requires the runtime driver.

## Interactive Viewer through a virtual desktop

For an interactive Viewer, also install `xvfb`, `x11vnc`, and a window manager
such as `fluxbox` or `xfce`, and expose display capability:

```Dockerfile
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
```

For example, with `fluxbox`:

```shell
x11vnc -create -env FD_PROG=/usr/bin/fluxbox \
  -env X11VNC_FINDDISPLAY_ALWAYS_FAILS=1 \
  -env X11VNC_CREATE_GEOM=${99:-1920x1080x16} \
  -gone 'pkill Xvfb' -nopw
```

Use a password and restrict network access in real deployments. After
connecting to the desktop on port 5900, run:

```shell
python -m sapien.example.hello_world
```
