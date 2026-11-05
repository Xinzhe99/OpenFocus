# Privacy Policy

OpenFocus does not upload your images, file names or project data. Desktop
usage reporting is enabled by default when no preference exists. An existing
explicitly disabled preference stays disabled. Change the switch in
**Settings → Advanced Settings → Share usage statistics**, then choose **OK**;
Cancel, Esc or closing the dialog discards the draft. There is no onboarding
choice or additional consent dialog.

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

## Data stored locally

Settings, recent-file lists, recovery data and image caches are stored on the
local machine (in portable mode, beside the executable). Image and project data
are not uploaded. When usage reporting is enabled, its random installation ID is
also stored locally and sent as described above.

Diagnostic logs stay local and rotate at approximately 1 MB per file with three
backups per process; inactive process logs expire after 14 days. Logs describe
processing stages, methods, resources, errors and cancellation timing, and may
contain source/output paths and final-pixel checksums. Usage reporting never
reads these logs. Feedback
packages are exported locally for review and manual submission; they contain
system facts, a bounded session excerpt and up to 20 MB of recent log tails
(at most 1 MB each and 40 files). A package index identifies omitted or truncated records.

## Contact

Open an issue at <https://github.com/Xinzhe99/OpenFocus/issues>.
