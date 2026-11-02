# OpenFocus installer for Windows.
#
#   irm https://openfocus.top/install.ps1 | iex
#
# Downloads the latest setup.exe from GitHub Releases and runs it.
# Set $env:OPENFOCUS_DRY_RUN = "1" first to print the plan without downloading.

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$site = "https://openfocus.top"
Write-Host "Reading the latest OpenFocus release..."
$release = Invoke-RestMethod -Uri "$site/api/latest"

$asset = $release.assets.'windows-setup'
if (-not $asset) { throw "No Windows installer found in $($release.tag)" }

$size = [math]::Round($asset.size / 1e6)
Write-Host "Latest release: $($release.tag)"
Write-Host "Download: $($asset.name) ($size MB)"
$url = "$site/download/latest/windows-setup"

if ($env:OPENFOCUS_DRY_RUN) {
    Write-Host "Dry run: not downloading."
    exit 0
}

$out = Join-Path $env:TEMP "OpenFocus-setup.exe"
Invoke-WebRequest -Uri $url -OutFile $out
Write-Host "Starting the installer..."
Start-Process -FilePath $out -Wait
