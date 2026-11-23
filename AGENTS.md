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

## Temporary workspace

Keep task-specific scripts, screenshots, fixtures, and test output in a named
folder under the checkout's ignored `.tmp/` directory. Put temporary Git
worktrees under `.tmp/worktrees/`. Use the application's configured scratch
directory for generated runtime files. Keep the drive root and workspace
parent clear of temporary files and folders.

When the task is complete, archive any results needed for recovery or review.
Remove the task's remaining temporary files and use `git worktree remove` to
retire its temporary worktrees after preserving any local changes.
