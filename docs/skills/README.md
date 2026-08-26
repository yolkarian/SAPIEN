# Quickstart with Agents

This directory helps users who build SAPIEN simulation environments get started fast *with an agent*. An agent reading these guides can quickly learn how to set up a SAPIEN simulation environment — GPU PhysX configuration, scene/build/render initialization order, direct and staged GPU Viewer rendering, batched rendering, the exposed PhysX GPU buffers, and where to find each API.

These are practical, lookup-oriented references (not tutorial prose): workflow rules the agent follows, plus a grep-friendly map of the SAPIEN Python API it can consult instead of guessing names or signatures.

## Contents

- `sapien-simulation/SKILL.md` — workflow checklist an agent follows when creating, refactoring, or reviewing a SAPIEN simulation environment.
- `sapien-simulation/docs/gpu-workflows.md` — GPU simulation workflows: PhysX GPU setup, env IDs, CUDA buffers, direct/staged Viewer submission and interaction, Docker Vulkan/EGL setup, and exposed PhysX GPU articulation link/Jacobian buffers.
- `sapien-simulation/docs/api/` — compact API tables covering core, scene/builders/loaders, PhysX CPU/GPU, rendering, sensors, math, and the low-level internal renderer/UI.

## How to use it

- Point your agent at `SKILL.md` and `gpu-workflows.md` first for the required initialization order and GPU rules, then let it use the API tables under `api/` as a fast lookup.
- Install SAPIEN from the fork's GitHub release wheels at `https://github.com/yolkarian/SAPIEN/releases`; do not install SAPIEN from PyPI.
- When the API tables and the checked-in source disagree, prefer the source/stubs/CI.

## Maintenance notes

- Treat `docs/skills/sapien-simulation/` as the source of truth. Every SAPIEN API addition, removal, rename, signature/default change, or behavior/lifecycle/ownership change must update the relevant workflow guidance and API tables, including `docs/api/api-changes.md`, in the same change.
- During local agent work, sync the checked-in skill to `~/.agents/skills/sapien-simulation/` after editing it only when that installed skill directory already exists; do not create it when absent, and never use it as the source.
- Keep content free of local machine details, private paths, secrets, tokens, and temporary absolute paths.
- Prefer repository-relative paths, environment variables, or placeholder names in examples.
- When syncing or adding a guide, adapt links, paths, commands, and examples to this repository and remove any local-only information before committing.
