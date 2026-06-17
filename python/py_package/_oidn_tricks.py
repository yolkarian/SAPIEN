import ctypes
import os
import platform
from pathlib import Path


_oidn_handles = []


def _load_first_existing(oidn_dir: Path, *patterns: str) -> None:
    for pattern in patterns:
        paths = sorted(oidn_dir.glob(pattern), reverse=True)
        for path in paths:
            if not path.is_file():
                continue
            try:
                _oidn_handles.append(ctypes.CDLL(str(path), ctypes.RTLD_LOCAL))
                return
            except OSError:
                continue


if platform.system() == "Linux":
    oidn_library_dir = Path(__file__).resolve().parent / "oidn_library"
    if oidn_library_dir.is_dir():
        _load_first_existing(
            oidn_library_dir,
            "libOpenImageDenoise_core.so.*",
            "libOpenImageDenoise_core.so",
        )
        _load_first_existing(
            oidn_library_dir,
            "libOpenImageDenoise.so.*",
            "libOpenImageDenoise.so",
        )
        _load_first_existing(
            oidn_library_dir,
            "libOpenImageDenoise_device_cuda.so.*",
            "libOpenImageDenoise_device_cuda.so",
        )

if platform.system() == "Windows":
    oidn_library_dir = Path(__file__).resolve().parent / "oidn_library"
    for library_name in ["OpenImageDenoise_core.dll", "OpenImageDenoise.dll"]:
        try:
            _oidn_handles.append(ctypes.CDLL(str(oidn_library_dir / library_name)))
        except OSError:
            pass
