# Project contribution guidance

Keep public project materials focused on OpenFocus behavior and documented
open-source dependencies. Do not name or compare proprietary competing focus
stacking products, or describe investigations of proprietary implementations.

Apply this rule to source code, comments, documentation, tests, fixture names,
commit messages, pull requests, issues, releases, and other public project
materials. Describe functionality and validation in OpenFocus terms.

## GitHub closeout

After verification, merge completed pull requests or close them with the
reason if they are superseded. Close resolved issues and link the merged fix
or other verification. Delete temporary remote branches promptly after their
changes are merged or preserved elsewhere. Keep branches and issues open
while work still depends on them.

## Pull request checks and end-of-day CI

Run focused local tests for behavior changes before pushing a pull request.
Use GitHub's required checks for merges. Prefer fast static checks for
routine PRs and track pending checks in the background while continuing
independent work.

Process new PRs when the maintainer requests a merge batch. After the
maintainer confirms the day's PRs have been handled, run the full test
matrix on `main` with `gh workflow run tests.yml --ref main -f run_suite=true`.
Investigate failures across platforms, verify fixes locally, and rerun the
matrix on the corrected `main` when needed. The release workflow runs the
full matrix against the exact release commit.

## Daily closeout

After the day's validation and fixes, review the repository and workspace.

1. Review local and remote branches. Remove surplus branches after their
   changes are merged or preserved and keep branches for active work.
2. Work both open queues, pull requests and issues, down to zero using the
   GitHub closeout rules. Record any unresolved item and its blocker.
3. Archive artifacts needed for review or recovery. Remove remaining
   temporary files and folders and retire temporary worktrees as described
   below.

## Temporary workspace

Keep task-specific scripts, screenshots, fixtures, and test output in a named
folder under the checkout's ignored `.tmp/` directory. Put temporary Git
worktrees under `.tmp/worktrees/`. Use the application's configured scratch
directory for generated runtime files. Keep the system drive, drive root,
workspace parent, and default OS user profile/temp directories (`%LOCALAPPDATA%`,
`%APPDATA%`, `%TEMP%`, and equivalent user caches) clear of temporary files,
isolated preview profiles, inference packs, build outputs, and tool caches.

When running tests, preview desktop builds, or packaging checks, explicitly
redirect `TEMP`, `TMP`, `OPENFOCUS_DATA_DIR`, and any package/build caches
(`PIP_CACHE_DIR`, `UV_CACHE_DIR`, `PYINSTALLER_CONFIG_DIR`, virtual/Conda
environments) to a task folder under `.tmp/` or a non-system workspace cache
directory.

When the task is complete, archive any results needed for recovery or review.
Remove the task's remaining temporary files, preview profiles, and downloaded
inference packs, and use `git worktree remove` to retire its temporary worktrees
after preserving any local changes.
