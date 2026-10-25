import contextlib
import io
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional
from zipfile import ZipFile

import requests

from ..pysapien.physx import *
from ..pysapien.physx import _enable_gpu


@contextlib.contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``path`` (created if missing) for the block."""
    with open(path, "a+b") as f:
        if platform.system() == "Windows":
            import msvcrt

            # LK_LOCK gives up after ~10 s; keep retrying while another process downloads.
            while True:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _find_library(candidates: List[Path]) -> Optional[Path]:
    return next((path for path in candidates if path.exists()), None)


def _download_library(url: str, parent: Path, candidates: List[Path]) -> Path:
    """Download and extract the PhysX GPU package into ``parent``.

    The archive is extracted into a private temporary directory and each top-level entry is
    then renamed into place, so a library path only appears once it is completely written.
    Callers hold the download lock, so no other process renames into ``parent`` meanwhile.
    """
    print(
        f"Downloading PhysX GPU library to {parent} from Github. This can take several minutes."
        f" If it fails to download, please manually download {url} and unzip at {parent}."
    )
    res = requests.get(url)
    res.raise_for_status()
    staging = Path(tempfile.mkdtemp(prefix=".extract-", dir=parent))
    try:
        ZipFile(io.BytesIO(res.content)).extractall(staging)
        for entry in staging.iterdir():
            target = parent / entry.name
            # Only a partial extraction from an interrupted pre-lock download can be here:
            # complete installs return before downloading.
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists() or target.is_symlink():
                target.unlink()
            os.replace(entry, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    dll = _find_library(candidates)
    if dll is None:
        raise RuntimeError(f"Downloaded PhysX GPU package does not contain expected library: {url}")
    print("Download complete.")
    return dll


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
        url = f"https://github.com/yolkarian/physx-release/releases/download/{physx_version}-Linux/physxgpu-linux-clang.zip"
    else:
        raise RuntimeError("Unsupported platform")

    dll = _find_library(candidates)
    if dll is None:
        # Multi-process jobs (torchrun, one process per GPU) share this cache and start
        # together: let one process download while the others wait, then reuse its install.
        with _exclusive_file_lock(parent / ".download.lock"):
            dll = _find_library(candidates) or _download_library(url, parent, candidates)

    import ctypes

    if platform.system() == "Windows":
        ctypes.CDLL("cuda.dll", ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(dll), ctypes.RTLD_LOCAL)
    elif platform.system() == "Linux":
        ctypes.CDLL("libcuda.so", ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(dll), ctypes.RTLD_LOCAL)

    _enable_gpu()
