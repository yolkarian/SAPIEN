---
name: sapien-release
description: Manually trigger SAPIEN tagged GitHub releases with gh after reviewing commits since the previous release tag, writing human-summarized release notes, and maintaining CHANGELOG.md.
disable-model-invocation: true
---

# SAPIEN Release

This skill is manual-only. Use `/skill:sapien-release` when preparing or
publishing a SAPIEN fork release through the manual GitHub Actions tagged
release workflow.

## Managed files

Paths below are relative to the repository root:

- `.github/workflows/build-tagged-release.yml`: workflow dispatched by `gh`.
- `CHANGELOG.md`: manually maintained project changelog.

Run shell commands from the repository root unless a command says otherwise.

## Release note style

Release notes follow the [pi coding agent](https://github.com/earendil-works/pi/releases) style: scannable, one line per entry, no internal mechanism detail.

### Template

```markdown
## Highlights

- **One-line user-facing summary of the biggest change.** One more sentence of context if needed. See [docs link](url).
- **Second highlight if applicable.** Keep to 2-4 entries at most.

## Breaking

- Removed `old_api()` — use `new_api()` instead. Migration: ...
- Changed `foo` default from `X` to `Y` (reason in one clause). Update by ...

## Added

- Added `NewClass` / `new_method()` for <purpose> ([#123](link))
- Added `config.new_field` (`type`, default `value`) — controls <what> ([#123](link))

## Changed

- Changed <what> to <new behavior> (one clause of why) ([#123](link))
- Deprecated `old` in favor of `new`; `old` still works until next breaking release

## Fixed

- Fixed <symptom> when <condition> ([#123](link))
- Fixed <crash/hang/data-loss> in <component> ([#123](link))

## Build and packaging

- Wheels now ... / Requires ...

See `CHANGELOG.md` for reasoning, measurements, and migration detail behind each entry.
```

### Rules

- **One entry = one line** (two at most). Lead with the observable effect, then the action the reader must take. Name the replacement for anything removed or renamed on the same line.
- **Highlights first.** Pick 1-4 entries a returning user would care about most. Each highlight is a bold summary plus one sentence and an optional docs link. Highlights may overlap with entries below — that is intentional.
- **Categories in order:** `Highlights` (optional, for notable releases), `Breaking`, `Added`, `Changed`, `Fixed`, `Build and packaging` (when packaging/toolchain changed), `Documentation`. Omit empty sections.
- **No mechanism, measurements, or file-level detail.** Move symbol names beyond the API surface, internal file paths, benchmark numbers, and design rationale to `CHANGELOG.md`; point there once in the footer rather than repeating it inline.
- **Drop entries with no user-visible or integrator-visible effect.** Refactors, comment cleanups, and internal renames do not appear unless they change behavior.
- **Link where possible.** Append `([#123](url))` / `([#123](url) by @user)` for traceability when an issue or PR exists.
- **Aim for well under 80 lines total.** If it reads like documentation, it is too long.
- `CHANGELOG.md` carries the same concise entries plus an extra paragraph of reasoning/evidence where it matters. It uses the same category headings so the two stay aligned.

### Before / after

Bad (old style — a paragraph per entry):

> Fixed indexed PhysX GPU APIs reading and writing outside SAPIEN and PhysX buffers for out-of-range `gpu_index` values, which silently corrupted state or caused `cudaErrorIllegalAddress` and poisoned the CUDA context. Caller-supplied index buffers are now range-checked on the configured SAPIEN CUDA stream, and rejected calls raise `RuntimeError` before anything is applied; the check synchronizes that stream once per indexed call (disable it with `PhysxSceneConfig.gpu_index_validation`), while non-indexed overloads are unchanged.

Good (pi style — one line, action on the same line):

> - Fixed out-of-range `gpu_index` values in indexed PhysX GPU APIs silently corrupting GPU state — now raises `RuntimeError` on the SAPIEN CUDA stream ([#123](link))

## Optimized release workflow

Use one preflight snippet to authenticate, fetch tags, choose defaults, inspect
the release range, and verify that the target commit is visible to GitHub. If
you later create a changelog/release-prep commit, rerun this snippet so
`GIT_REF` still points at the intended final commit.

1. Run preflight and inspect the release range:

   ```bash
   gh auth status
   git fetch origin
   mapfile -t TAG_REFS < <(git ls-remote --tags --refs origin | awk '$2 ~ /^refs\/tags\/[0-9]/ { print $2 ":" $2 }')
   if ((${#TAG_REFS[@]})); then
     git fetch origin "${TAG_REFS[@]}"
   fi

   if [ -z "${GIT_REF:-}" ]; then
     GIT_REF=$(git rev-parse HEAD)
     export GIT_REF
   fi
   export WORKFLOW_REF=${WORKFLOW_REF:-$(git symbolic-ref --quiet --short HEAD || echo dev)}
   export PRERELEASE=${PRERELEASE:-true}

   if [ -z "${RELEASE_TAG:-}" ]; then
     LAST_RELEASE_TAG=$(git for-each-ref --sort=-v:refname --format='%(refname:short)' refs/tags \
       | grep -E '^[0-9]' \
       | head -1)
     if [ -z "${LAST_RELEASE_TAG}" ]; then
       echo "No previous numeric release tag found; set RELEASE_TAG explicitly." >&2
       exit 1
     fi
     RELEASE_TAG=$(python -c 'import re, sys; tag = sys.argv[1]; matches = list(re.finditer(r"\d+", tag)); m = matches[-1]; print(f"{tag[:m.start()]}{int(m.group()) + 1}{tag[m.end():]}")' "${LAST_RELEASE_TAG}")
     export RELEASE_TAG
   fi

   if git rev-parse --verify --quiet "refs/tags/${RELEASE_TAG}" >/dev/null \
       || git ls-remote --exit-code --tags --refs origin "${RELEASE_TAG}" >/dev/null 2>&1; then
     echo "Release tag already exists: ${RELEASE_TAG}" >&2
     exit 1
   fi

   TARGET_COMMIT=$(git rev-parse "${GIT_REF}^{commit}")
   PREVIOUS_TAG=$(git for-each-ref --merged "${TARGET_COMMIT}" --sort=-v:refname --format='%(refname:short)' refs/tags \
     | grep -E '^[0-9]' \
     | grep -vx "${RELEASE_TAG}" \
     | head -1)
   if [ -n "${PREVIOUS_TAG}" ]; then
     RANGE="${PREVIOUS_TAG}..${TARGET_COMMIT}"
   else
     RANGE="${TARGET_COMMIT}"
   fi

   if ! git ls-remote --exit-code --heads --tags origin "${GIT_REF}" >/dev/null 2>&1 \
       && ! git branch -r --contains "${TARGET_COMMIT}" | grep -q .; then
     echo "Target commit is not visible on fetched remote refs; push it before dispatch." >&2
     exit 1
   fi

   printf 'git ref: %s\nworkflow ref: %s\nrelease tag: %s\nprerelease: %s\nprevious tag: %s\ntarget commit: %s\nrange: %s\n' \
     "${GIT_REF}" "${WORKFLOW_REF}" "${RELEASE_TAG}" "${PRERELEASE}" "${PREVIOUS_TAG:-<none>}" "${TARGET_COMMIT}" "${RANGE}"
   git log --reverse --oneline "${RANGE}"
   if [ -n "${PREVIOUS_TAG}" ]; then
     git diff --stat "${RANGE}"
   else
     git show --stat --summary "${TARGET_COMMIT}"
   fi
   ```

   Defaults:

   - `GIT_REF`: current `HEAD` commit unless the user clearly specifies a commit,
     branch, or ref to release.
   - `WORKFLOW_REF`: current branch, falling back to `dev` for detached HEAD.
   - `PRERELEASE`: `true`.
   - `RELEASE_TAG`: latest numeric release tag with its final number incremented
     by one, for example `3.0.0+fork.7` -> `3.0.0+fork.8`.

2. Read and analyze commits before writing release notes.

   The preflight log and stat are usually enough for small, clear commits. For
   vague, large, or user-facing commits, inspect details before summarizing:

   ```bash
   git show --stat --summary <commit-sha>
   git show --name-only --format=fuller <commit-sha>
   git diff "${PREVIOUS_TAG}..${TARGET_COMMIT}" -- <important-path>
   ```

   Write release notes using the template and rules in [Release note style](#release-note-style). Keep them scannable — one line per entry, observable effect first, replacement named on the same line. Save mechanism and measurements for `CHANGELOG.md`.

3. Write release notes and update the changelog if needed:

   ```bash
   export RELEASE_NOTES_FILE=${RELEASE_NOTES_FILE:-$(mktemp)}
   "$EDITOR" "$RELEASE_NOTES_FILE"
   sed -n '1,180p' "$RELEASE_NOTES_FILE"
   ```

   Update `CHANGELOG.md` before dispatch when the release includes changes not
   yet recorded there. Use the same concise category headings as the release
   notes (`Highlights` / `Breaking` / `Added` / `Changed` / `Fixed` / `Build and packaging`). Rename the standing `## Unreleased` heading to a dated `RELEASE_TAG` section, then add a fresh empty `## Unreleased` above it so the
   next change has somewhere to land. Commit and push that release-preparation change, then rerun step 1 so `GIT_REF`, `TARGET_COMMIT`, and `RANGE` reflect the final pushed commit.

4. Dispatch the workflow with the reviewed notes:

   ```bash
   python -c 'import json, os; from pathlib import Path; print(json.dumps({"git_ref": os.environ["GIT_REF"], "release_tag": os.environ["RELEASE_TAG"], "prerelease": os.environ.get("PRERELEASE", "true"), "release_notes": Path(os.environ["RELEASE_NOTES_FILE"]).read_text()}))' \
     | gh workflow run build-tagged-release.yml --ref "${WORKFLOW_REF}" --json
   ```

   `gh workflow run --json` expects workflow input values as strings. Keep
   `prerelease` as the string `true` or `false`; do not JSON-encode it as a
   boolean.

5. Report the run URL/status without waiting for the long build by default:

   ```bash
   gh run list --workflow build-tagged-release.yml --limit 5
   ```

   Only watch the run when the user explicitly asks:

   ```bash
   gh run watch <run-id> --exit-status
   ```

6. Once the release is published (`gh release view "${RELEASE_TAG}"` succeeds), update the
   locally installed simulation skill from the release tag, only if that directory exists.
   This is the only time the installed copy changes; during development only the
   checked-in `docs/skills/sapien-simulation/` follows the code.

   ```bash
   skill_dir="$HOME/.agents/skills/sapien-simulation"
   if [ -d "$skill_dir" ]; then
     tmp=$(mktemp -d)
     git archive "${RELEASE_TAG}" docs/skills/sapien-simulation | tar -x -C "$tmp"
     rsync -a --delete --exclude .DS_Store "$tmp/docs/skills/sapien-simulation/" "$skill_dir/"
     rm -rf "$tmp"
   fi
   ```

## Guardrails

- This skill is manual-only; invoke it explicitly with `/skill:sapien-release`.
- The workflow only consumes `release_notes`; it must not generate final release notes itself.
- Always read/analyze commits since the previous release tag before composing release notes.
- Do not publish raw commit logs as release descriptions unless the user explicitly asks.
- Do not paste changelog sections into release notes verbatim — rewrite to one line per entry per the template above.
- Do not overwrite or delete an existing GitHub release or stable release tag without explicit user approval.
- Ignore `nightly` when computing the previous release tag; it is a moving test release.
- If the user does not clearly specify the release commit/ref (`GIT_REF`), use the current `HEAD` commit, not `dev`, `main`, or another moving branch.
- Before dispatch, ensure the target commit is pushed and visible from remote refs.
- If a changelog or release-prep commit is created, rerun preflight and release the new final `HEAD`.
- If the user does not specify `RELEASE_TAG`, use the latest numeric release tag and increment its final number by one.
- Prefer version tags that start with a digit, such as `3.0.0+fork.8`.
- Pass boolean workflow inputs to `gh workflow run --json` as strings.
- Do not watch long release builds by default; report the run URL/status and wait only when asked.
- Keep `CHANGELOG.md` curated and human-readable. Use the same category headings as release notes.
- Leave a standing empty `## Unreleased` section at the top of `CHANGELOG.md` after every release.
