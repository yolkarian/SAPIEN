# Changelog

Notable user-facing and developer-facing changes are documented here.
Release descriptions are written from reviewed commits and diffs, then passed to the tagged release workflow as input.

## Unreleased

### Added

- Added a required Markdown release notes input to the tagged release workflow.
- Added a manual-only project-level Pi skill for reviewing commits since the previous release tag, composing summarized release notes, triggering tagged releases through `gh`, and maintaining this changelog.
- Added default release-tag selection that increments the final number from the latest numeric release tag unless a version is specified.
- Added sanitized convenience copies of agent skills under `docs/skills` for repository-local use.

### Changed

- Limited push-triggered nightly builds to the `main` branch.
