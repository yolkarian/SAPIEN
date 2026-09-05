# Testing GPU stream ordering

The regression suite in `unittest/test_physx/test_gpu_stream_order.py` exercises
SAPIEN's CUDA stream contract without a policy, training framework, robot assets,
or renderer. It requires CUDA-enabled Torch and GPU PhysX. Torch supplies test
buffers, delayed CUDA kernels, and tensor assertions; physics uses SAPIEN alone.

From the repository root, run the asynchronous case first:

```bash
env -u CUDA_LAUNCH_BLOCKING PYTHONPATH=unittest/test_physx \
  python -m unittest -v test_gpu_stream_order
```

Then run a diagnostic control in a fresh process:

```bash
CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=unittest/test_physx \
  python -m unittest -v test_gpu_stream_order
```

For native error investigation, select one test per process, for example:

```bash
PYTHONPATH=unittest/test_physx python -m unittest -v \
  test_gpu_stream_order.TestGpuStreamOrder.test_articulation_fetch_preserves_queued_consumers
```

## What is checked

- **Rigid-body and articulation-link pose snapshots:** fetch a reference, queue
  a delayed fetch and tensor clone, step physics, then fetch again. The earlier
  clone must equal the reference, not the next step's state. Both default and
  non-default CUDA streams are tested with 32 bodies/articulations.
- **Joint buffer ordering:** poison fetched qpos/qvel buffers on the configured
  stream, then fetch and clone them without an intervening host wait. Fetch must
  replace the poison before the clone reads the data.
- **Joint-first state-fetch sequence:** qpos, qvel, qacc, link pose, link velocity,
  and rigid-body fetches run in a typical batched RL order. This control matters
  because earlier operations can incidentally order an otherwise unsafe fetch.
- **Contact-query/step ordering:** delay a cached, nonzero asynchronous contact
  query and immediately step. The queued snapshot must still contain the prior
  step's impulses.
- **Contact disappearance:** move all boxes away from the ground using GPU state
  APIs. Subsequent contact queries must return zeros rather than old impulses.

The delay kernels intentionally enlarge legal overlap between the configured
SAPIEN stream and PhysX's internal non-blocking streams. They are not throughput
benchmarks. Checks synchronize only after the sequence under test has been
submitted; adding a wait between every operation would mask the defect.

## Scratch-buffer ordering fix

Rigid-body data and articulation-link pose fetches previously supplied only a
PhysX completion event. That ordered conversion after the new copy, but did not
prevent a later PhysX copy from overwriting scratch still needed by an earlier
conversion. These paths now also record a start event on the configured SAPIEN
stream and pass it to PhysX. This preserves asynchronous submission and avoids
requiring `CUDA_LAUNCH_BLOCKING=1` or a device-wide barrier.

A reproducible wrong-snapshot result demonstrates a stream-ordering bug. It does
not, by itself, identify the cause of a separate driver/native crash. Likewise,
a passing contact or joint-first control only establishes the tested cases; it
is not proof that every mixed simulation/rendering workload is race-free.
