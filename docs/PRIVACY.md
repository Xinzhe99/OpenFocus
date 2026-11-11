# Privacy Policy

OpenFocus does not upload your images, file names or project data. Desktop
usage reporting is enabled by default when no preference exists. An existing
explicitly disabled preference stays disabled. Change the switch in
**Settings → Advanced Settings → Share usage statistics**, then choose **OK**;
Cancel, Esc or closing the settings dialog discards the draft. Usage statistics
are independent of account sign-in and diagnostic reporting.

## Network access

- **Update checks and release notes** — the desktop app requests release metadata
  and notes from `openfocus.top` automatically (at most once per day) or when
  the user checks manually. Background checks are quiet. Requests include
  standard HTTP headers and network information such as the IP address.
- **Update download** — a supported desktop build may automatically download a
  newer release from `openfocus.top` while idle; installation requires the user
  to choose to restart and update. Builds without self-update open the website
  download page instead.
- **Usage reporting** — while enabled, the desktop app sends only a
  randomly generated installation ID, app version and operating system to
  `openfocus.top`, normally no more than once every 24 hours after a successful
  report. Failed sends are retried while the application remains open. The ID is
  pseudonymous, not anonymous or derived from hardware. Images, paths, project
  data, hardware details and diagnostic logs are not part of usage reporting.
  Turning the setting off deletes the local ID and send timestamp and stops
  future reports; enabling it again creates a new ID. Closing the application
  also cancels future reports. A request already in flight may still complete.
- **Optional account** — email, GitHub or Google sign-in uses your browser and
  `account.openfocus.top`. The account service receives the verified email and
  identity needed for sign-in. You can skip sign-in, sign out, or delete your
  account. Authentication credentials are stored locally and are not included in
  diagnostic reports. Signing out also requests server session revocation. If
  local credential removal fails, the account dialog reports incomplete cleanup
  and offers a retry; signing out in memory does not erase credentials on disk.
  The service also receives a random ID saved for this profile and a device
  label based on the computer name when exchanging a sign-in code, to manage
  the three-device limit. Account requests include the app version, operating
  system, and a random ID and start time for the current app run.
- **Error reports** — you can review and send a sanitized report, signed in or
  anonymously. After desktop sign-in, automatic error reporting is enabled by
  default unless you previously disabled it; a saved preference is preserved.
  You can change it in Advanced Settings. Automatic reports are prepared only
  after a session records an error; a previous native crash is offered for review
  and requires a separate confirmation. Automatic reports are skipped for
  development runs; report metadata includes the build channel.
  Reports can include a separate random
  report installation ID,
  software and system versions, selected hardware details, image counts and
  dimensions, the order of supported interface actions, control IDs and canvas
  gesture coordinates, effective processing options and numeric settings, error
  types and selected application stack locations. Native crash reports can include
  allowlisted OpenFocus file and function names and line numbers, but not raw
  crash text, memory addresses, user file names or paths. Reports exclude image
  pixels, image and project files, credentials and arbitrary user-entered text.
  Reports retain at most the most recent 2,000 recognized events from a bounded
  log tail and indicate detected omissions; older rotated logs may already be gone.
  Failed reports can be stored locally for bounded retries, with at most three
  pending reports and three attempts. Account-linked queued reports are retried
  only with the same account. Turning automatic reporting off revokes permission
  for requests that have not yet been authorized to start and removes queued
  automatic reports. A request already authorized to start may still complete;
  previously received reports are not immediately deleted. Manually submitted
  reports may still be retried. Queue transactions coordinate processes sharing
  the same local profile; a setting change is not broadcast to other running
  processes.
- **Website downloads** — when download analytics is enabled on the server,
  the site records aggregate download starts by date, version, asset type,
  route and source. Retries, scanners and incomplete downloads may be counted.
  These counts do not identify unique users or completed downloads.
- **Headless CLI** — makes no network requests.

The headless CLI makes no usage reports. Network requests include standard HTTP
headers and network information such as the IP address. Cloudflare may retain
its own request logs independently of the analytics database.

## Analytics retention

The analytics database stores daily usage records for up to 90 UTC calendar
days; operational failures can delay deletion. Turning off usage reporting does
not immediately remove records already received. Aggregated download counts
are retained until an operator deletes them. See the [website operations
guide](WEBSITE_OPERATIONS.md#optional-private-analytics) for details.

Error-report bundles are deleted after 90 days without activity; report index
records and events are deleted after 180 days. Account deletion removes reports
linked to that account. Anonymous reports use the independent report installation
ID and follow the same retention schedule.

## Data stored locally

Settings, recent-file lists, recovery data and image caches are stored on the
local machine (in portable mode, beside the executable). Image and project data
are not uploaded. When usage reporting is enabled, its random installation ID is
also stored locally and sent as described above.

Diagnostic logs remain on the local machine unless you choose to share them;
they rotate at approximately 1 MB per file with three backups per process, and
inactive process logs expire after 14 days. Local logs may contain source/output
paths and final-pixel checksums. Usage statistics never read these logs. A local
feedback export may include up to 20 MB of recent log tails and should be reviewed
before sharing. The separate error-report bundle contains only the sanitized
fields described above; raw logs and local feedback exports are not uploaded.
If the log directory is unavailable, new native crash files use a private,
profile-specific temporary directory and follow the same inactive-process
retention period. Older crash files left directly in the system temporary
directory cannot be attributed safely to a profile and are not discovered
automatically; they can be located and exported manually for support.

## Contact

Open an issue at <https://github.com/Xinzhe99/OpenFocus/issues>.
