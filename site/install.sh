#!/bin/sh
# OpenFocus installer for macOS and Linux.
#
#   curl -fsSL https://openfocus.top/install.sh | sh
#   curl -fsSL https://openfocus.top/install.sh | sh -s -- --dry-run
#
# macOS: installs the latest Apple Silicon build into ~/Applications.
# Linux: installs the headless CLI from PyPI (there is no Linux GUI build).
# Windows: prints the PowerShell command instead.

set -eu

SITE="https://openfocus.top"
API="$SITE/api/latest"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

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
  [ "$DRY" = 1 ] && { say "Would install the CLI: pipx install --force openfocus"; exit 0; }
  if command -v pipx >/dev/null 2>&1; then
    say "Installing the OpenFocus CLI with pipx..."
    pipx install --force openfocus
  elif command -v python3 >/dev/null 2>&1; then
    say "Installing the OpenFocus CLI with pip..."
    python3 -m pip install --user --upgrade openfocus
  else
    fail "install python3 and pip, then run: python3 -m pip install --user openfocus"
  fi
  say "Done. Run: openfocus --help"
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
[ "$DRY" = 1 ] && exit 0

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
say "Downloading..."
curl -fL --progress-bar "$url" -o "$tmp/OpenFocus.zip"
unzip -q "$tmp/OpenFocus.zip" -d "$tmp"

app=$(find "$tmp" -maxdepth 3 -type d -name 'OpenFocus.app' | head -1)
[ -n "$app" ] || fail "OpenFocus.app was not found in the archive"

mkdir -p "$HOME/Applications"
rm -rf "$HOME/Applications/OpenFocus.app"
cp -R "$app" "$HOME/Applications/"
xattr -cr "$HOME/Applications/OpenFocus.app" 2>/dev/null || true

say "Installed: $HOME/Applications/OpenFocus.app"
say "The app is unsigned: on first launch use right-click → Open if macOS blocks it."
open "$HOME/Applications/OpenFocus.app" 2>/dev/null || true
