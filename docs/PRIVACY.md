# Privacy Policy

OpenFocus does not collect telemetry, analytics or personal data, and it does
not upload user data to any server.

## Network access

- **Update check** — once per day the desktop app requests release metadata from
  the GitHub Releases API (`api.github.com/repos/Xinzhe99/OpenFocus`). The
  request contains no account or user identifier beyond the standard HTTP
  headers; GitHub's privacy policy applies to that request.
- **Update download** — only when the user chooses to update, the app downloads
  the release asset from GitHub or `openfocus.top`.
- **Headless CLI** — makes no network requests.

No other outbound requests are made by the application.

## Data stored locally

Settings, recent-file lists, recovery data and image caches are stored on the
local machine (in portable mode, beside the executable). They are never
uploaded.

## Contact

Open an issue at <https://github.com/Xinzhe99/OpenFocus/issues>.
