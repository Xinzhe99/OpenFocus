# Join OpenFocus Windows split release assets on Windows PowerShell 5.1+.
#
# Download the manifest (OpenFocus-<tag>-windows-assets.json), every *.partNNN
# file listed in it, and this script into one folder, then run:
#   powershell -ExecutionPolicy Bypass -File .\join-openfocus-parts.ps1 `
#       -Manifest .\OpenFocus-<tag>-windows-assets.json
#
# Each part and the joined file are checked against the SHA-256 values in the
# manifest. A failed check removes the temporary output, leaving any existing
# file untouched.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Manifest,
    [string]$OutputDirectory = "."
)

$ErrorActionPreference = "Stop"

function Get-Sha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    try {
        $sha = [Security.Cryptography.SHA256]::Create()
        try {
            return [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
        }
        finally {
            $sha.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}
$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$sourceDirectory = Split-Path -Parent $manifestPath
$data = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($data.schema -ne 1) {
    throw "Unsupported manifest schema: $($data.schema)"
}
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) {
    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
}

foreach ($property in $data.assets.PSObject.Properties) {
    $asset = $property.Value
    $parts = @()
    if ($null -ne $asset.parts) {
        $parts = @($asset.parts)
    }
    if ($parts.Count -eq 0) {
        Write-Host "Whole asset (no parts): $($asset.name)"
        continue
    }

    $target = Join-Path $OutputDirectory $asset.name
    if ([IO.Path]::GetFileName($target) -ne $asset.name) {
        throw "Unsafe output name: $($asset.name)"
    }
    $temporary = Join-Path $OutputDirectory ("{0}.{1}.tmp" -f $asset.name, [guid]::NewGuid().ToString('N'))
    try {
        $output = [IO.File]::Create($temporary)
        try {
            foreach ($part in $parts) {
                if ([IO.Path]::GetFileName($part.name) -ne $part.name) {
                    throw "Unsafe part name: $($part.name)"
                }
                $partPath = Join-Path $sourceDirectory $part.name
                if (-not (Test-Path -LiteralPath $partPath -PathType Leaf)) {
                    throw "Missing part: $($part.name)"
                }
                $item = Get-Item -LiteralPath $partPath
                if ($item.Length -ne [int64]$part.size) {
                    throw "Part size mismatch: $($part.name)"
                }
                $partHash = Get-Sha256 $partPath
                if ($partHash -ne ([string]$part.sha256).ToLowerInvariant()) {
                    throw "Part SHA-256 mismatch: $($part.name)"
                }
                $reader = [IO.File]::OpenRead($partPath)
                try {
                    $reader.CopyTo($output)
                }
                finally {
                    $reader.Dispose()
                }
            }
        }
        finally {
            $output.Dispose()
        }

        $joined = Get-Item -LiteralPath $temporary
        if ($joined.Length -ne [int64]$asset.size) {
            throw "Joined size mismatch: $($asset.name)"
        }
        $joinedHash = Get-Sha256 $temporary
        if ($joinedHash -ne ([string]$asset.sha256).ToLowerInvariant()) {
            throw "Joined SHA-256 mismatch: $($asset.name)"
        }
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            $backup = "$temporary.backup"
            [IO.File]::Replace($temporary, $target, $backup)
            Remove-Item -LiteralPath $backup -Force -ErrorAction SilentlyContinue
        }
        else {
            [IO.File]::Move($temporary, $target)
        }
        Write-Host "Created $target"
        Write-Host "SHA-256 $joinedHash"
    }
    finally {
        if (Test-Path -LiteralPath $temporary -PathType Leaf) {
            Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
        }
    }
}
