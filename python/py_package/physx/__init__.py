import io
import platform
from pathlib import Path
from zipfile import ZipFile

import requests

from ..pysapien.physx import *
from ..pysapien.physx import _enable_gpu


def enable_gpu():
    if is_gpu_enabled():
        return

    physx_version = version()
    parent = Path.home() / ".sapien" / "physx" / physx_version
    parent.mkdir(exist_ok=True, parents=True)

    candidates = []
    if platform.system() == "Windows":
        candidates = [
            parent / "PhysXGpu_64.dll",
            parent
            / "physxgpu-windows-vc17win64"
            / "PhysX"
            / "bin"
            / "win.x86_64.vc143.mt"
            / "release"
            / "PhysXGpu_64.dll",
        ]
        url = f"https://github.com/yolkarian/physx-release/releases/download/{physx_version}-windows/physxgpu-windows-vc17win64.zip"
    elif platform.system() == "Linux":
        candidates = [
            parent / "libPhysXGpu_64.so",
            parent
            / "physxgpu-linux-clang"
            / "PhysX"
            / "bin"
            / "linux.x86_64"
            / "release"
            / "libPhysXGpu_64.so",
        ]
        url = f"https://github.com/yolkarian/physx-release/releases/download/{physx_version}/physxgpu-linux-clang.zip"
    else:
        raise RuntimeError("Unsupported platform")

    dll = next((path for path in candidates if path.exists()), candidates[-1])
    if not dll.exists():
        print(
            f"Downloading PhysX GPU library to {parent} from Github. This can take several minutes."
            f" If it fails to download, please manually download {url} and unzip at {parent}."
        )
        res = requests.get(url)
        res.raise_for_status()
        z = ZipFile(io.BytesIO(res.content))
        z.extractall(parent)
        dll = next((path for path in candidates if path.exists()), None)
        if dll is None:
            raise RuntimeError(f"Downloaded PhysX GPU package does not contain expected library: {url}")
        print("Download complete.")

    import ctypes

    if platform.system() == "Windows":
        ctypes.CDLL("cuda.dll", ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(dll), ctypes.RTLD_LOCAL)
    elif platform.system() == "Linux":
        ctypes.CDLL("libcuda.so", ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(dll), ctypes.RTLD_LOCAL)

    _enable_gpu()
