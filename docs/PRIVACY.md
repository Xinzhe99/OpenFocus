# Privacy Policy

OpenFocus does not collect telemetry, analytics or personal data, and it does
not upload user data to any server.

## Network access

- **Update checks and release notes** — the desktop app requests release metadata
  and notes from `openfocus.top` automatically (at most once per day) or when
  the user checks manually. Background checks are quiet. Requests include
  standard HTTP headers and network information such as the IP address.
- **Update download** — a supported desktop build may automatically download a
  newer release from `openfocus.top` while idle; installation requires the user
  to choose to restart and update. Builds without self-update open the website
  download page instead.
- **Headless CLI** — makes no network requests.

No other outbound requests are made by the application.

## Data stored locally

Settings, recent-file lists, recovery data and image caches are stored on the
local machine (in portable mode, beside the executable). They are never
uploaded.

## Contact

Open an issue at <https://github.com/Xinzhe99/OpenFocus/issues>.
