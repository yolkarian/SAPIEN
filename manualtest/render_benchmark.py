"""Throughput of grouped offscreen rendering (RenderSystemGroup) on one GPU.

Each configuration runs in its own process, because shader caches are process-wide and GPU
memory should be measured from a clean start. A configuration builds `num_envs` scenes with
`objects` static primitives each and one camera per scene, then times
`RenderSystemGroup.update_render()` + `RenderCameraGroup.take_picture()`.

`take_picture()` returns once the frame is submitted; it first waits for the previous frame's
copy into the exported buffers, so the loop measures steady-state throughput with one frame in
flight. A final extra capture drains the last frame before the clock stops.

    python render_benchmark.py --out results.csv            # default matrix
    python render_benchmark.py --shaders rt --envs 64 256 --sizes 128 --out rt.csv
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

FIELDS = [
    "shader", "color_format", "num_envs", "objects", "width", "height", "frames",
    "images_per_s", "ms_per_frame", "update_ms", "capture_ms", "init_s", "gpu_mem_mib", "status",
]


def gpu_memory_mib() -> float:
    """GPU memory held by this process according to nvidia-smi, or -1 when unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return -1.0
    for line in out.splitlines():
        pid, _, used = line.partition(",")
        if pid.strip() == str(os.getpid()):
            return float(used)
    return -1.0


def run_worker(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the scenes of one configuration, render, and return its measurements."""
    import numpy as np
    import sapien

    render_config = sapien.render.RenderConfig(config["shader"])
    if config["shader"] == "rt":
        render_config["spp"] = config["spp"]
        render_config["maxDepth"] = config["depth"]
        render_config.denoiser = "none"
    elif config["shader"] == "rt_batch":
        # batched ray tracing: the pack renders the whole camera group in one dispatch, so the
        # scene only needs a RenderSystem per environment
        render_config["spp"] = config["spp"]
    if config["defines"]:
        render_config.defines = dict(pair.split("=", 1) if "=" in pair else (pair, "")
                                     for pair in config["defines"])
    if config["color_format"] != "default":
        render_config.picture_formats = {"Color": config["color_format"]}
    sapien.render.set_default_render_config("camera", render_config)

    start = time.perf_counter()
    rng = np.random.default_rng(0)
    scenes, cameras = [], []
    materials = [
        sapien.render.RenderMaterial(base_color=[*rng.uniform(0.2, 0.9, 3), 1.0]) for _ in range(8)
    ]
    for _ in range(config["num_envs"]):
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem()])
        scene.set_ambient_light([0.3, 0.3, 0.3])
        scene.add_directional_light([0.3, 0.2, -1.0], [2.0, 2.0, 2.0], shadow=config["shadows"])
        scene.add_ground(0.0)
        for index in range(config["objects"]):
            builder = scene.create_actor_builder()
            position = [*rng.uniform(-0.6, 0.6, 2), 0.1 + 0.15 * (index % 4)]
            if index % 2:
                builder.add_sphere_visual(radius=0.08, material=materials[index % 8])
            else:
                builder.add_box_visual(half_size=[0.07, 0.07, 0.07], material=materials[index % 8])
            builder.build_kinematic().set_pose(sapien.Pose(position))
        camera = scene.add_camera(
            "camera",
            config["width"],
            config["height"],
            1.0,
            0.05,
            20.0,
            render_config=render_config,
        )
        # above the table, looking down at 45 degrees
        camera.entity.pose = sapien.Pose([-1.6, 0.0, 1.6], [0.9238795, 0.0, 0.3826834, 0.0])
        scenes.append(scene)
        cameras.append(camera)

    group = sapien.render.RenderSystemGroup([scene.render_system for scene in scenes])
    camera_group = group.create_camera_group(cameras, ["Color"])
    group.gpu_init()

    def frame() -> tuple[float, float]:
        t0 = time.perf_counter()
        group.update_render()
        t1 = time.perf_counter()
        camera_group.take_picture()
        return t1 - t0, time.perf_counter() - t1

    for _ in range(config["warmup"]):
        frame()
    init_s = time.perf_counter() - start

    update_s = capture_s = 0.0
    t0 = time.perf_counter()
    for _ in range(config["frames"]):
        update, capture = frame()
        update_s += update
        capture_s += capture
    frame()  # waits for the last timed frame
    elapsed = time.perf_counter() - t0
    frames = config["frames"]
    picture = camera_group.get_picture_cuda("Color")
    result = {
        "images_per_s": round(frames * config["num_envs"] / elapsed, 1),
        "ms_per_frame": round(1e3 * elapsed / frames, 3),
        "update_ms": round(1e3 * update_s / frames, 3),
        "capture_ms": round(1e3 * capture_s / frames, 3),
        "init_s": round(init_s, 2),
        "gpu_mem_mib": gpu_memory_mib(),
        "status": f"ok {picture.typestr} {tuple(picture.shape)}",
    }
    del picture
    group.close()
    for scene in scenes:
        scene.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--out", default="render_benchmark.csv")
    parser.add_argument("--shaders", nargs="+", default=["default", "minimal", "rt"])
    parser.add_argument("--envs", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--objects", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--color-formats", nargs="+", default=["default", "r8g8b8a8unorm"],
                        help="Color target formats for raster packs; 'default' is float32")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--spp", type=int, default=1, help="ray-tracing samples per pixel")
    parser.add_argument("--defines", nargs="*", default=[],
                        help="shader defines as NAME or NAME=VALUE, e.g. SHADOWS")
    parser.add_argument("--depth", type=int, default=2, help="ray-tracing path depth")
    parser.add_argument("--shadows", action="store_true", help="directional light casts shadows")
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per configuration")
    args = parser.parse_args()

    if args.worker:
        print("RESULT " + json.dumps(run_worker(json.loads(args.worker))), flush=True)
        return

    configs: List[Dict[str, Any]] = []
    for shader in args.shaders:
        formats = ["default"] if shader == "rt" else args.color_formats
        for color_format in formats:
            for objects in args.objects:
                for size in args.sizes:
                    for num_envs in args.envs:
                        configs.append({
                            "shader": shader, "color_format": color_format, "num_envs": num_envs,
                            "objects": objects, "width": size, "height": size,
                            "frames": args.frames, "warmup": args.warmup, "spp": args.spp,
                            "depth": args.depth, "shadows": args.shadows,
                            "defines": args.defines,
                        })

    with open(args.out, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for config in configs:
            row = dict(config)
            try:
                process = subprocess.run(
                    [sys.executable, __file__, "--worker", json.dumps(config)],
                    capture_output=True, text=True, timeout=args.timeout,
                )
                lines = [l for l in process.stdout.splitlines() if l.startswith("RESULT ")]
                if process.returncode == 0 and lines:
                    row.update(json.loads(lines[-1][len("RESULT "):]))
                else:
                    tail = (process.stderr or process.stdout).strip().splitlines()[-1:]
                    row["status"] = f"failed({process.returncode}): {' '.join(tail)}"[:300]
            except subprocess.TimeoutExpired:
                row["status"] = f"timeout after {args.timeout:.0f}s"
            writer.writerow(row)
            file.flush()
            print({key: row.get(key) for key in FIELDS}, flush=True)


if __name__ == "__main__":
    main()
