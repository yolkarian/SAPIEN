# Agent Skills

This directory contains convenience copies of agent skills that are useful when working with this repository.

These files are not SAPIEN runtime documentation. They are lightweight workflow guides for coding agents and maintainers, kept in the repository so they can be reviewed, shared, and used without relying on a local machine-specific skills checkout.

## Included skills

- `sapien-simulation`: SAPIEN simulation workflow guidance, including GPU PhysX setup, rendering, sensors, reset/step order, batched rendering, and container notes.

## Maintenance notes

- Keep skill content free of local machine details, private paths, secrets, tokens, and temporary absolute paths.
- Prefer repository-relative paths, environment variables, or placeholder names in examples.
- When adding or syncing a skill, ask an agent to rewrite it for this repository instead of copying it verbatim.
- During that rewrite, adapt links, paths, commands, and examples to this repository and remove any local-only information.
- Review the copied diff before committing and sanitize any local-only information.
