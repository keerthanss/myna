<#
    Removes Myna. Offers to keep the downloaded models, which are
    several gigabytes and live outside the program folder.
#>

[CmdletBinding()]
param(
    [switch]$KeepModels,
    [switch]$Silent
)

$ErrorActionPreference = 'Stop'
$AppName = 'Myna'
$InstallDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Models normally live inside the install folder, so removing it removes them
# too. Only a folder the user relocated needs separate handling.
$DataDir = Join-Path $InstallDir 'data'
$ModelsDir = $DataDir
$configPath = Join-Path $DataDir 'config.json'
if (Test-Path $configPath) {
    try {
        $configured = (Get-Content $configPath -Raw | ConvertFrom-Json).models_root
        if ($configured -and -not $configured.StartsWith($InstallDir)) { $ModelsDir = $configured }
    } catch { }
}

if (-not $Silent) {
    Write-Host ''
    Write-Host "  Uninstall $AppName" -ForegroundColor White
    Write-Host "  Program folder: $InstallDir" -ForegroundColor DarkGray
    Write-Host ''
    $answer = Read-Host '  Remove the application? [y/N]'
    if ($answer -notmatch '^[Yy]') {
        Write-Host '  Cancelled.' -ForegroundColor Yellow
        exit 0
    }
    if (-not $KeepModels -and $ModelsDir -ne $DataDir -and (Test-Path $ModelsDir)) {
        $size = (Get-ChildItem $ModelsDir -Recurse -File -ErrorAction SilentlyContinue |
                 Measure-Object -Property Length -Sum).Sum / 1GB
        $modelAnswer = Read-Host ("  Models were moved outside the app folder. Also delete them ({0:N1} GB) at {1}? [y/N]" -f $size, $ModelsDir)
        if ($modelAnswer -notmatch '^[Yy]') { $KeepModels = $true }
    }
}

Write-Host ''
Write-Host '==> Removing shortcuts' -ForegroundColor Cyan
$shortcuts = @(
    (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\$AppName.lnk"),
    (Join-Path ([Environment]::GetFolderPath('Desktop')) "$AppName.lnk")
)
foreach ($shortcut in $shortcuts) {
    if (Test-Path $shortcut) { Remove-Item $shortcut -Force }
}

Write-Host '==> Removing the Add/Remove Programs entry' -ForegroundColor Cyan
$uninstallKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Myna'
if (Test-Path $uninstallKey) { Remove-Item $uninstallKey -Recurse -Force }

# Older builds kept preferences in the registry; current ones use an INI file
# inside the app folder. Clean up the legacy key if it is still there.
$settingsKey = 'HKCU:\Software\WhisperStudio'
if (Test-Path $settingsKey) {
    Write-Host '==> Removing legacy registry preferences' -ForegroundColor Cyan
    Remove-Item $settingsKey -Recurse -Force
}

if (-not $KeepModels -and $ModelsDir -ne $DataDir -and (Test-Path $ModelsDir)) {
    Write-Host '==> Removing relocated models' -ForegroundColor Cyan
    Remove-Item $ModelsDir -Recurse -Force -ErrorAction SilentlyContinue
} elseif ($ModelsDir -ne $DataDir -and (Test-Path $ModelsDir)) {
    Write-Host "==> Keeping models at $ModelsDir" -ForegroundColor Cyan
}

# The program folder holds this script, so delete it from a detached process.
Write-Host '==> Removing the program folder' -ForegroundColor Cyan
$command = "Start-Sleep -Seconds 2; Remove-Item -LiteralPath '$InstallDir' -Recurse -Force -ErrorAction SilentlyContinue"
Start-Process powershell.exe -ArgumentList '-NoProfile', '-WindowStyle', 'Hidden', '-Command', $command

Write-Host ''
Write-Host "  $AppName has been uninstalled." -ForegroundColor Green
Write-Host ''
if (-not $Silent) { Start-Sleep -Seconds 3 }
