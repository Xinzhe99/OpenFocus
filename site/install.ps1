# OpenFocus installer for Windows.
#
#   irm https://openfocus.top/install.ps1 | iex
#
# Downloads the latest setup.exe through openfocus.top (the R2 mirror, with a
# GitHub release fallback) and runs it.
# Set $env:OPENFOCUS_DRY_RUN = "1" (or "true") first to print the plan without
# downloading or installing anything.

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

$dryRun = $env:OPENFOCUS_DRY_RUN -match '^(1|true|yes|on)$'
if ($dryRun) {
    Write-Host "Dry run: would download $url and run the setup. Nothing was downloaded or installed."
    exit 0
}

$out = Join-Path $env:TEMP "OpenFocus-$($release.tag -replace '[^A-Za-z0-9._-]', '_')-setup.exe"
Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
if ($asset.size -and (Get-Item -LiteralPath $out).Length -ne [int64]$asset.size) {
    Remove-Item -LiteralPath $out -Force -ErrorAction SilentlyContinue
    throw "The downloaded OpenFocus installer is incomplete."
}
if ($asset.digest -match '^sha256:([0-9a-fA-F]{64})$') {
    $expectedHash = $matches[1].ToLowerInvariant()
    $actualHash = (Get-FileHash -LiteralPath $out -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedHash) {
        Remove-Item -LiteralPath $out -Force -ErrorAction SilentlyContinue
        throw "The downloaded OpenFocus installer failed SHA-256 verification."
    }
}
Write-Host "Starting the installer..."
$process = Start-Process -FilePath $out -Wait -PassThru
if ($process.ExitCode -ne 0) {
    throw "The OpenFocus installer exited with code $($process.ExitCode)."
}
