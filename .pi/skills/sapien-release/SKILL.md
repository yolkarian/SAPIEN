---
name: sapien-release
description: Trigger SAPIEN tagged GitHub releases with gh after reviewing commits since the previous release tag, writing human-summarized release notes, and maintaining CHANGELOG.md.
---

# SAPIEN Release

Use this skill when preparing or publishing a SAPIEN fork release through the
manual GitHub Actions tagged release workflow.

## Managed files

Paths below are relative to this skill directory:

- `../../../.github/workflows/build-tagged-release.yml`: workflow dispatched by `gh`.
- `../../../CHANGELOG.md`: manually maintained project changelog.

Run shell commands from the repository root unless a command says otherwise.

## Release workflow

1. Confirm `gh` is installed and authenticated for the target repository:

   ```bash
   gh auth status
   ```

2. Confirm the release target is pushed and fetch remote release tags. Do not
   rely on the moving `nightly` tag for release-note ranges:

   ```bash
   git fetch origin
   mapfile -t TAG_REFS < <(git ls-remote --tags --refs origin | awk '$2 ~ /^refs\/tags\/[0-9]/ { print $2 ":" $2 }')
   ((${#TAG_REFS[@]} == 0)) || git fetch origin "${TAG_REFS[@]}"
   ```

3. Choose release inputs:

   - `RELEASE_TAG`: package/release tag, e.g. `3.0.0+fork.8`; it must be a valid Python package version.
   - `GIT_REF`: branch, tag, or commit SHA to build, e.g. `dev` or a full commit SHA.
   - `WORKFLOW_REF`: branch containing the current workflow file, usually `dev`.
   - `PRERELEASE`: `true` for fork/internal releases unless the user asks otherwise.

4. Identify the previous release tag and inspect the change range:

   ```bash
   export RELEASE_TAG=3.0.0+fork.8
   export GIT_REF=dev
   TARGET_COMMIT=$(git rev-parse "${GIT_REF}^{commit}")
   PREVIOUS_TAG=$(git for-each-ref --merged "${TARGET_COMMIT}" --sort=-creatordate --format='%(refname:short)' refs/tags \
     | grep -E '^[0-9]' \
     | grep -vx "${RELEASE_TAG}" \
     | head -1)
   if [ -n "${PREVIOUS_TAG}" ]; then
     RANGE="${PREVIOUS_TAG}..${TARGET_COMMIT}"
   else
     RANGE="${TARGET_COMMIT}"
   fi
   printf 'previous tag: %s\ntarget commit: %s\nrange: %s\n' "${PREVIOUS_TAG:-<none>}" "${TARGET_COMMIT}" "${RANGE}"
   git log --reverse --oneline "${RANGE}"
   if [ -n "${PREVIOUS_TAG}" ]; then
     git diff --stat "${RANGE}"
   else
     git show --stat --summary "${TARGET_COMMIT}"
   fi
   ```

5. Read and analyze the commits before writing release notes.

   Use commit messages as a starting point, but do not publish a raw script-generated list. For vague, large, or user-facing commits, inspect details:

   ```bash
   git show --stat --summary <commit-sha>
   git show --name-only --format=fuller <commit-sha>
   git diff "${PREVIOUS_TAG}..${TARGET_COMMIT}" -- <important-path>
   ```

   Summarize the release in human-facing Markdown. Prefer grouped bullets such as `Highlights`, `Fixes`, `Build and packaging`, `Documentation`, and `Validation` when applicable. Mention commit hashes only when they help traceability. Do not include private paths, temporary files, or unrelated local validation details.

6. Write the summarized release notes to a temporary file and review them:

   ```bash
   export RELEASE_NOTES_FILE
   RELEASE_NOTES_FILE=$(mktemp)
   "$EDITOR" "$RELEASE_NOTES_FILE"
   sed -n '1,160p' "$RELEASE_NOTES_FILE"
   ```

7. Update `CHANGELOG.md` before publishing when the release includes changes not yet recorded there. Add a dated section for the release tag and summarize notable user-facing or developer-facing changes. Keep the changelog curated and concise; it does not need to mirror the full release description.

8. Commit and push any release-preparation changes that must be included in the target ref before dispatching the workflow. Do not release from a local-only commit unless the user explicitly confirms.

9. Trigger the tagged release workflow with the reviewed release notes as the release description:

   ```bash
   export WORKFLOW_REF=dev
   export PRERELEASE=true
   python -c 'import json, os; from pathlib import Path; print(json.dumps({"git_ref": os.environ["GIT_REF"], "release_tag": os.environ["RELEASE_TAG"], "prerelease": os.environ.get("PRERELEASE", "true").lower() == "true", "release_notes": Path(os.environ["RELEASE_NOTES_FILE"]).read_text()}))' \
     | gh workflow run build-tagged-release.yml --ref "${WORKFLOW_REF}" --json
   ```

10. Watch the run and report the result:

   ```bash
   gh run list --workflow build-tagged-release.yml --limit 5
   gh run watch <run-id> --exit-status
   ```

## Guardrails

- The workflow only consumes `release_notes`; it must not generate final release notes itself.
- Always read/analyze commits since the previous release tag before composing release notes.
- Do not publish raw commit logs as release descriptions unless the user explicitly asks.
- Do not overwrite or delete an existing GitHub release or stable release tag without explicit user approval.
- Ignore `nightly` when computing the previous release tag; it is a moving test release.
- Prefer version tags that start with a digit, such as `3.0.0+fork.8`.
- Keep `CHANGELOG.md` curated and human-readable.
