# Privacy Policy

OpenFocus does not upload your images, file names or project data. Optional
desktop usage reporting is disabled by default and requires your consent.

## Network access

- **Update checks and release notes** — the desktop app requests release metadata
  and notes from `openfocus.top` automatically (at most once per day) or when
  the user checks manually. Background checks are quiet. Requests include
  standard HTTP headers and network information such as the IP address.
- **Update download** — a supported desktop build may automatically download a
  newer release from `openfocus.top` while idle; installation requires the user
  to choose to restart and update. Builds without self-update open the website
  download page instead.
- **Usage reporting** — if you enable it in Settings, the desktop app sends a
  randomly generated installation ID, app version and operating system to
  `openfocus.top`, normally no more than once every 24 hours after a successful
  report. The ID is pseudonymous, not anonymous or derived from hardware.
  Turning the setting off deletes the local ID and stops future reports;
  a request already in flight may still complete.
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

## Contact

Open an issue at <https://github.com/Xinzhe99/OpenFocus/issues>.
