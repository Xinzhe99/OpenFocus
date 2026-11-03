#!/bin/sh
# OpenFocus installer for macOS and Linux.
#
#   curl -fsSL https://openfocus.top/install.sh | sh
#   curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
#
# macOS: installs the latest Apple Silicon build into ~/Applications. The new
#        app is staged and checked first, then swapped in; the previous app is
#        restored if the swap fails.
# Linux: installs the headless CLI from PyPI (there is no Linux GUI build) with
#        pipx or a private virtual environment. System Python is never modified,
#        including on PEP 668 (externally managed) distributions.
# Windows: prints the PowerShell command instead.

set -eu

SITE="https://openfocus.top"
API="$SITE/api/latest"
DRY=0

case "${1:-}" in
  "") ;;
  --dry-run) DRY=1 ;;
  *)
    printf 'OpenFocus: unknown argument: %s (supported: --dry-run)\n' "$1" >&2
    exit 2
    ;;
esac

say() { printf '%s\n' "$*"; }
fail() { printf 'OpenFocus: %s\n' "$*" >&2; exit 1; }

os=$(uname -s 2>/dev/null || echo unknown)
arch=$(uname -m 2>/dev/null || echo unknown)
say "OpenFocus installer — $os/$arch"

case "$os" in
  MINGW*|MSYS*|CYGWIN*)
    say "Windows detected. Run this in PowerShell instead:"
    say "  irm https://openfocus.top/install.ps1 | iex"
    exit 0
    ;;
  Darwin) ;;
  Linux) ;;
  *) fail "unsupported platform: $os" ;;
esac

if [ "$os" = "Linux" ]; then
  # pipx is preferred: it is already an isolated environment and works on
  # distributions that mark the system Python as externally managed (PEP 668).
  if command -v pipx >/dev/null 2>&1; then
    if [ "$DRY" = 1 ]; then
      say "Would install the CLI in an isolated pipx environment:"
      say "  pipx install --force openfocus"
      exit 0
    fi
    say "Installing the OpenFocus CLI with pipx..."
    pipx install --force openfocus \
      || fail "pipx could not install OpenFocus. Check the network and pipx's own configuration, then retry."
    say "Done. Run: openfocus --help"
    say "If the command is not found, add $HOME/.local/bin to PATH."
    exit 0
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    if [ "$DRY" = 1 ]; then
      say "python3 is not installed."
      say "Install Python 3.10+ and pipx, then rerun this installer."
      exit 0
    fi
    fail "Python 3.10+ is required. Install python3 and pipx (https://pipx.pypa.io/stable/installation/), then rerun."
  fi

  if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    python_version=$(python3 -V 2>&1 || true)
    if [ "$DRY" = 1 ]; then
      say "Found $python_version; OpenFocus requires Python 3.10 or newer."
      exit 0
    fi
    fail "Python 3.10+ is required (found $python_version)."
  fi

  if ! python3 -c 'import ensurepip, venv' >/dev/null 2>&1; then
    if [ "$DRY" = 1 ]; then
      say "python3 is present but its venv/ensurepip support is missing."
      say "Install pipx (recommended) or your distribution's python3-venv package, then rerun."
      exit 0
    fi
    fail "cannot create an isolated environment (venv/ensurepip missing). Install pipx (https://pipx.pypa.io/stable/installation/) or your distribution's python3-venv package, then rerun. No system-wide pip install is attempted."
  fi

  base="$HOME/.local/share/openfocus"
  venv="$base/venv"
  backup="$base/venv.previous.$(date +%Y%m%d%H%M%S).$$"
  linkdir="$HOME/.local/bin"
  link="$linkdir/openfocus"
  linux_mutating=0
  linux_installed=0

  linux_cleanup() {
    if [ "$linux_installed" = 1 ]; then
      return 0
    fi
    if [ "$linux_mutating" != 1 ]; then
      return 0
    fi
    rm -rf "$venv" 2>/dev/null || true
    if [ -e "$backup" ]; then
      mv "$backup" "$venv" 2>/dev/null \
        || printf 'OpenFocus: previous CLI environment preserved at %s\n' "$backup" >&2
    fi
  }
  trap linux_cleanup EXIT
  trap 'exit 1' HUP INT TERM

  if [ "$DRY" = 1 ]; then
    say "Would create an isolated virtual environment:"
    say "  python3 -m venv $venv"
    say "  $venv/bin/python -m pip install --upgrade openfocus"
    say "Would link $link to $venv/bin/openfocus."
    say "System Python is not modified."
    exit 0
  fi

  say "Installing the OpenFocus CLI in an isolated virtual environment..."
  mkdir -p "$base" || fail "cannot create $base"
  if [ -e "$venv" ] || [ -L "$venv" ]; then
    mv "$venv" "$backup" || fail "cannot preserve the existing CLI environment at $venv"
  fi
  linux_mutating=1

  if ! python3 -m venv "$venv"; then
    fail "could not create the virtual environment at $venv"
  fi

  if ! "$venv/bin/python" -m pip install --upgrade openfocus; then
    fail "pip could not install OpenFocus in the isolated environment. Check the network, then retry."
  fi

  if [ ! -x "$venv/bin/openfocus" ]; then
    fail "the openfocus command was not installed in $venv/bin"
  fi

  linux_installed=1
  if [ -e "$backup" ]; then
    rm -rf "$backup" || say "warning: could not remove the previous environment at $backup"
  fi

  mkdir -p "$linkdir" || say "warning: cannot create $linkdir"
  if [ -e "$link" ] && [ ! -L "$link" ]; then
    say "Installed: $venv/bin/openfocus"
    say "Not replacing the existing file $link (it is not a symlink). Move it aside, then run:"
    say "  ln -sfn \"$venv/bin/openfocus\" \"$link\""
  elif ln -sfn "$venv/bin/openfocus" "$link" 2>/dev/null; then
    say "Installed: $link"
    case ":${PATH:-}:" in
      *":$linkdir:"*)
        say "Done. Run: openfocus --help"
        ;;
      *)
        say "Add $linkdir to PATH, then run: openfocus --help"
        say "  export PATH=\"$linkdir:\$PATH\""
        ;;
    esac
  else
    say "Installed: $venv/bin/openfocus"
    say "Could not create $link. Add $venv/bin to PATH or use the full path above."
  fi
  exit 0
fi

# macOS
command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v unzip >/dev/null 2>&1 || fail "unzip is required"
[ "$arch" = "arm64" ] || fail "macOS builds are Apple Silicon only (this machine reports $arch)"

say "Reading the latest release..."
json=$(curl -fsSL "$API") || fail "cannot reach $SITE"
tag=$(printf '%s' "$json" | sed -n 's/.*"tag": *"\([^"]*\)".*/\1/p' | head -1)
[ -n "$tag" ] || fail "cannot read the latest release tag"

url="$SITE/download/latest/macos-zip"
say "Latest release: $tag"
say "Download: $url"
say "Install to: $HOME/Applications/OpenFocus.app"

if [ "$DRY" = 1 ]; then
  say "Would download the archive, verify the app bundle, stage a complete copy,"
  say "then replace the existing app. A failed replacement restores the previous app."
  exit 0
fi

tmp=$(mktemp -d) || fail "cannot create a temporary directory"
dest="$HOME/Applications/OpenFocus.app"
stage_dir=""
stage=""
backup=""

cleanup() {
  [ -n "${tmp:-}" ] && rm -rf "$tmp" 2>/dev/null || true
  [ -n "${stage:-}" ] && rm -rf "$stage" 2>/dev/null || true
  if [ -n "${backup:-}" ] && { [ -e "$backup" ] || [ -L "$backup" ]; } && \
     [ ! -e "$dest" ] && [ ! -L "$dest" ]; then
    mv "$backup" "$dest" 2>/dev/null || true
  fi
  # Keep the private directory if it still contains a backup that could not
  # be restored; otherwise remove only the empty directory we created.
  [ -n "${stage_dir:-}" ] && rmdir "$stage_dir" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

say "Downloading..."
curl -fL --progress-bar "$url" -o "$tmp/OpenFocus.zip" || fail "download failed"
unzip -q "$tmp/OpenFocus.zip" -d "$tmp" || fail "cannot unpack the archive"

app=$(find "$tmp" -maxdepth 3 -type d -name 'OpenFocus.app' | head -1)
[ -n "$app" ] || fail "OpenFocus.app was not found in the archive"

mkdir -p "$HOME/Applications" || fail "cannot create $HOME/Applications"

# Stage a complete copy beside the destination (same filesystem) before
# touching the installed app.
stage_dir=$(mktemp -d "$HOME/Applications/.OpenFocus.install.XXXXXXXX") \
  || fail "cannot create a staging directory in $HOME/Applications"
stage="$stage_dir/OpenFocus.app"
say "Staging the new app..."
if command -v ditto >/dev/null 2>&1; then
  ditto "$app" "$stage" || fail "could not stage the new app; the existing installation was not changed"
else
  cp -R "$app" "$stage" || fail "could not stage the new app; the existing installation was not changed"
fi

if [ ! -f "$stage/Contents/Info.plist" ] || [ ! -x "$stage/Contents/MacOS/OpenFocus" ]; then
  fail "the staged app is incomplete; the existing installation was not changed"
fi

# The build is not notarized; clear quarantine on the staged copy before it moves.
xattr -cr "$stage" 2>/dev/null || true

had_previous=0
if [ -e "$dest" ] || [ -L "$dest" ]; then
  had_previous=1
  backup="$stage_dir/previous.app"
  mv "$dest" "$backup" || fail "cannot move the existing app aside; it was not changed"
fi

if ! mv "$stage" "$dest"; then
  if [ "$had_previous" = 1 ] && { [ -e "$backup" ] || [ -L "$backup" ]; }; then
    if mv "$backup" "$dest"; then
      backup=""
      fail "could not install the new app; the previous installation was restored"
    fi
    fail "could not install the new app or restore the previous one. The previous app is preserved at: $backup"
  fi
  fail "could not install the new app; no previous installation was present"
fi

stage=""
if [ "$had_previous" = 1 ] && { [ -e "$backup" ] || [ -L "$backup" ]; }; then
  rm -rf "$backup" || say "warning: could not remove the old app backup at $backup"
  backup=""
fi

say "Installed: $dest"
say "The app is not notarized. If macOS blocks it, use System Settings → Privacy & Security → Open Anyway."
open "$dest" 2>/dev/null || true
