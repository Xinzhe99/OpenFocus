# Engineering lessons

This file records reusable lessons from work on OpenFocus. Add a lesson when it
can change a future engineering decision, not for each incident or release.
Keep entries short and actionable; revise or remove them when the code or
operating assumptions change. Detailed requirements and procedures belong in the
relevant source, tests, or documentation linked from [CONTRIBUTING.md](CONTRIBUTING.md).

## Evidence and verification

- Separate an observation from its cause. A crash timestamp, a passing retry,
  or an enabled provider does not establish the failing thread or where work
  ran. Preserve diagnostics and state what remains unknown.
- A crash handler can fail while reporting an earlier fault. Resolve the
  recorded address before treating the handler's module as the original cause.
- Verify the path users actually exercise. For GUI changes, cover the rendered
  controls and relevant cross-process interactions; for accelerators, observe
  execution and fallback as well as provider availability.
- Test the boundary that caused a failure: delayed or out-of-order worker
  completion, a closed dialog, a contended lock, an unavailable service, or a
  failed allocation. Keep tests isolated and assert visible behavior.

## State, lifetime, and data

- Tie asynchronous results to the state they describe. When an operation is
  superseded, cancel or ignore its late callbacks, and ensure the current state
  still receives a completion or refresh.
- Keep workers, callbacks, and dialogs alive only as long as their owners need
  them. Cancel timers with their owner and wait for native work to finish before
  destroying resources it still uses.
- Isolate test profiles, runtime caches, and temporary files off the system drive
  and default OS user directories by pointing `TEMP`, `TMP`,
  `OPENFOCUS_DATA_DIR`, and tool caches onto the workspace volume or `.tmp/`.
  Clean up preview profiles, downloaded inference packs, and session-owned
  artifacts on completion; preserve failure evidence and leave files of unknown
  ownership alone.
- Apply access and feature gates consistently across desktop controls, batch
  workflows, and the control channel. Preserve the user's saved state and make
  unavailable or offline behavior explicit.

## Delivery

- Make release channels and their defaults explicit in both client and service.
  Test version selection, missing metadata, and cached or offline cases together.
- Integrate against the current main branch without overwriting unrelated work.
  Reviewable changes should contain the intended behavior, focused verification,
  and no unrelated formatting or line-ending churn.
