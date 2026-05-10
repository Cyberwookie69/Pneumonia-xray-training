<#
.SYNOPSIS
Pull Colab run artefacts from Google Drive into a dated local folder.

.DESCRIPTION
Uses `gdown` (a wget-equivalent for Google Drive) to download the
shared `pneumonia_runs` folder. Filters out heavy .pt checkpoints by
default — only summary.json / history.json / .npy / .png / .jpg are
kept, which is what the report needs.

The Drive folder share-link is read from `.colab_drive_url` in the
project root (one-time setup; gitignored). Folder must be shared as
"Anyone with the link, viewer".

.PARAMETER TargetDir
Subdirectory under the project root (default: colab_output_YYYY-MM-DD).

.PARAMETER IncludeCheckpoints
Keep .pt model checkpoints (~100 MB each). Off by default.

.PARAMETER UrlOverride
Skip the .colab_drive_url file and use this URL directly.

.EXAMPLE
    .\_helpers\pull_colab_runs.ps1
    .\_helpers\pull_colab_runs.ps1 -TargetDir colab_run_03
    .\_helpers\pull_colab_runs.ps1 -IncludeCheckpoints
#>

param(
    [string]$TargetDir = "colab_output_$(Get-Date -Format 'yyyy-MM-dd')",
    [switch]$IncludeCheckpoints,
    [string]$UrlOverride = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

# Resolve the Drive URL
if ($UrlOverride) {
    $url = $UrlOverride
} else {
    $urlFile = Join-Path $projectRoot ".colab_drive_url"
    if (-not (Test-Path $urlFile)) {
        Write-Host "ERROR: $urlFile not found." -ForegroundColor Red
        Write-Host ""
        Write-Host "One-time setup:"
        Write-Host "  1. Open https://drive.google.com → My Drive → pneumonia_runs"
        Write-Host "  2. Right-click → Share → 'Anyone with the link' → 'Viewer'"
        Write-Host "  3. Copy the link (e.g. https://drive.google.com/drive/folders/1AbCxYz...)"
        Write-Host "  4. Save it to:"
        Write-Host "     $urlFile"
        Write-Host ""
        Write-Host "Or pass -UrlOverride <link> directly."
        exit 1
    }
    $url = (Get-Content $urlFile -Raw).Trim()
}

# Ensure gdown is installed
$gdownInstalled = $false
try {
    python -m pip show gdown 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { $gdownInstalled = $true }
} catch { }

if (-not $gdownInstalled) {
    Write-Host "Installing gdown..." -ForegroundColor Cyan
    python -m pip install gdown
}

# Create target directory
$targetPath = Join-Path $projectRoot $TargetDir
New-Item -ItemType Directory -Force -Path $targetPath | Out-Null
Write-Host "Target: $targetPath" -ForegroundColor Cyan
Write-Host "Source: $url" -ForegroundColor Cyan
Write-Host ""

# Download
python -m gdown --folder $url -O $targetPath --remaining-ok

# Filter out checkpoints unless requested
if (-not $IncludeCheckpoints) {
    $checkpoints = Get-ChildItem -Path $targetPath -Recurse -Include "*.pt", "*.pth", "*.ckpt" -ErrorAction SilentlyContinue
    if ($checkpoints) {
        $totalMB = ($checkpoints | Measure-Object -Property Length -Sum).Sum / 1MB
        Write-Host ""
        Write-Host "Removing $($checkpoints.Count) checkpoint files (~$([math]::Round($totalMB, 1)) MB)..." -ForegroundColor Yellow
        $checkpoints | Remove-Item -Force
        Write-Host "(Use -IncludeCheckpoints to keep them next time.)"
    }
}

# Summary
$kept = Get-ChildItem -Path $targetPath -Recurse -File
$totalMB = ($kept | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ""
Write-Host "Done. $($kept.Count) files kept, $([math]::Round($totalMB, 1)) MB." -ForegroundColor Green
Write-Host "Folder: $targetPath"
