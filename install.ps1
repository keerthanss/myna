<#
    Myna installer.

    Installs the app into one self-contained folder with its own
    private Python environment, then creates Start Menu and Desktop shortcuts
    and an Add/Remove Programs entry. Nothing is installed system-wide and no
    administrator rights are needed.
#>

[CmdletBinding()]
param(
    [switch]$CpuOnly,
    [switch]$NoShortcuts,

    # Everything - the app, its Python environment, models and caches - goes
    # inside this one folder, so it can live on any drive and be removed in one
    # go. Nothing is written to the system drive.
    [string]$InstallDir = 'D:\VibeCodedApps\Myna',

    # Models default to <InstallDir>\data\models. Override only to keep them
    # on a different drive from the app. Changeable later inside the app.
    [string]$ModelsDir = ''
)

$ErrorActionPreference = 'Stop'
$AppName = 'Myna'
$SourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Step($message) {
    Write-Host ''
    Write-Host "==> $message" -ForegroundColor Cyan
}

function Write-Note($message) {
    Write-Host "    $message" -ForegroundColor DarkGray
}

function Test-PythonCandidate {
    <#
        Returns the interpreter path when it is a usable version, else $null.
        Native tools here (notably py.exe) chatter on stderr, so errors are
        deliberately swallowed rather than allowed to stop the script.
    #>
    param([string]$Exe, [string[]]$PreArgs = @())

    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $script = "import sys;print('%d.%d.%d' % sys.version_info[:3]);print(sys.executable)"
        $output = & $Exe @PreArgs -c $script 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $output) { return $null }

        $lines = @($output | Where-Object { $_ -and $_.ToString().Trim() })
        if ($lines.Count -lt 2) { return $null }

        $parsed = [version]($lines[0].ToString().Trim())
        $found = $lines[1].ToString().Trim()
        if ($parsed -ge [version]'3.9' -and $parsed -lt [version]'3.13') {
            return [pscustomobject]@{ Path = $found; Version = $parsed }
        }
        Write-Note "Skipping Python $parsed at $found (needs 3.9-3.12)."
        return $null
    } catch {
        return $null
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Find-Python {
    # Prefer the newest supported interpreter; PySide6, ctranslate2 and the
    # llama-cpp-python wheels all publish builds for 3.9 through 3.12.
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        foreach ($version in @('3.12', '3.11', '3.10', '3.9')) {
            $result = Test-PythonCandidate -Exe $launcher.Source -PreArgs @("-$version")
            if ($result) { return $result }
        }
    }

    foreach ($name in @('python', 'python3')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            $result = Test-PythonCandidate -Exe $command.Source
            if ($result) { return $result }
        }
    }

    throw @'
No usable Python was found.

Myna needs Python 3.9 - 3.12. Install Python 3.12 from
https://www.python.org/downloads/windows/ and tick "Add python.exe to PATH"
during setup, then run this installer again.
'@
}

function New-Shortcut($LinkPath, $TargetPath, $Arguments, $WorkingDirectory, $IconPath, $Description) {
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($LinkPath)
    $shortcut.TargetPath = $TargetPath
    $shortcut.Arguments = $Arguments
    $shortcut.WorkingDirectory = $WorkingDirectory
    $shortcut.Description = $Description
    if ($IconPath -and (Test-Path $IconPath)) { $shortcut.IconLocation = $IconPath }
    $shortcut.Save()
}

Write-Host ''
Write-Host '  Myna - installer' -ForegroundColor White
Write-Host '  Local transcription, speaker labels and summaries' -ForegroundColor DarkGray
Write-Host ''

# ---------------------------------------------------------------- 1. Python
Write-Step 'Looking for Python'
$found = Find-Python
$python = $found.Path
Write-Note "Using Python $($found.Version) at $python"

# ------------------------------------------------------- 2. Check the target
Write-Step "Installing to $InstallDir"
$targetRoot = [System.IO.Path]::GetPathRoot($InstallDir)
if (-not (Test-Path $targetRoot)) {
    throw "The drive $targetRoot does not exist. Re-run with -InstallDir pointing somewhere that does, for example: .\Install.bat -InstallDir C:\Apps\Myna"
}
$targetFree = (Get-PSDrive -Name $targetRoot.TrimEnd(':') -ErrorAction SilentlyContinue).Free / 1GB
if ($targetFree -and $targetFree -lt 4) {
    Write-Warning ("Only {0:N1} GB free on {1}. The app and its packages need about 4 GB, plus 3-5 GB for models." -f $targetFree, $targetRoot)
}
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

# Source and install folder are allowed to be the same directory - that is the
# normal layout here, one folder holding both the code and the environment. In
# that case there is nothing to copy, and attempting it would be destructive:
# the folder-replace below deletes the destination first, which would delete
# the very source it is about to copy from.
$sourceResolved = (Resolve-Path -LiteralPath $SourceDir).Path.TrimEnd('\')
$installResolved = (Resolve-Path -LiteralPath $InstallDir).Path.TrimEnd('\')

if ($sourceResolved -ieq $installResolved) {
    Write-Note 'Running from the install folder itself - no files to copy.'
} else {
    foreach ($item in @('app', 'tools', 'assets', 'requirements.txt', 'README.md', 'uninstall.ps1')) {
        $source = Join-Path $SourceDir $item
        if (-not (Test-Path $source)) { continue }
        $destination = Join-Path $InstallDir $item
        if ((Get-Item $source).PSIsContainer) {
            # Replace the folder outright; Copy-Item -Recurse would otherwise
            # nest a second copy inside an existing destination on reinstall.
            if (Test-Path $destination) { Remove-Item $destination -Recurse -Force }
            Copy-Item -Path $source -Destination $destination -Recurse -Force
        } else {
            Copy-Item -Path $source -Destination $destination -Force
        }
    }
    Write-Note 'Application files copied.'
}

# ---------------------------------------------------------------- 3. Venv
$venv = Join-Path $InstallDir 'venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'
$venvPythonw = Join-Path $venv 'Scripts\pythonw.exe'

if (-not (Test-Path $venvPython)) {
    Write-Step 'Creating a private Python environment'
    & $python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment.' }
} else {
    Write-Step 'Reusing the existing Python environment'
}

# pip caches downloaded wheels under %LOCALAPPDATA% by default, which would
# put roughly 2 GB back on the system drive. Keep it inside the app folder.
$env:PIP_CACHE_DIR = Join-Path $InstallDir 'data\pip-cache'
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR | Out-Null

& $venvPython -m pip install --upgrade pip wheel --disable-pip-version-check --quiet
Write-Note 'pip is up to date.'
Write-Note "Wheel cache: $env:PIP_CACHE_DIR"

# ---------------------------------------------------------------- 4. Packages
Write-Step 'Installing packages (this downloads about 2 GB the first time)'
& $venvPython -m pip install --disable-pip-version-check -r (Join-Path $InstallDir 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Package installation failed. Scroll up for the pip error.' }

# ---------------------------------------------------------------- 5. llama.cpp
Write-Step 'Installing the summarisation engine (llama-cpp-python)'

# Version matters here. The current wheels (0.3.35 and nearby) are compiled with
# AVX-512, which Intel pre-Skylake-X and every AMD Zen 1-3 chip lack: the process
# dies with 0xc000001d (illegal instruction) at context creation, before the GPU
# is ever used. 0.3.4 is built for AVX2 and runs everywhere we have tested, with
# CUDA offload intact. Raise this only after checking a non-AVX-512 machine.
$llamaVersion = '0.3.4'

$llamaInstalled = $false
$indexes = @()
if (-not $CpuOnly) {
    $indexes += @{ Url = 'https://abetlen.github.io/llama-cpp-python/whl/cu124'; Label = 'CUDA 12.4 build (GPU accelerated)' }
    $indexes += @{ Url = 'https://abetlen.github.io/llama-cpp-python/whl/cu122'; Label = 'CUDA 12.2 build (GPU accelerated)' }
}
$indexes += @{ Url = 'https://abetlen.github.io/llama-cpp-python/whl/cpu'; Label = 'CPU build' }

foreach ($index in $indexes) {
    Write-Note "Trying $llamaVersion, $($index.Label)..."
    & $venvPython -m pip install --disable-pip-version-check --quiet "llama-cpp-python==$llamaVersion" --prefer-binary --only-binary=:all: --extra-index-url $index.Url
    if ($LASTEXITCODE -eq 0) {
        Write-Note "Installed $llamaVersion, $($index.Label)."
        $llamaInstalled = $true
        break
    }
}

if (-not $llamaInstalled) {
    Write-Warning 'Could not install llama-cpp-python. Transcription, speaker labels and subtitles will work; the summary PDFs will not.'
    Write-Warning 'You can retry later by running this installer again.'
}

# ---------------------------------------------------------------- 6. Icon
Write-Step 'Generating the application icon'
$iconPath = Join-Path $InstallDir 'assets\Myna.ico'
& $venvPython (Join-Path $InstallDir 'tools\make_icon.py') $iconPath
if ($LASTEXITCODE -ne 0) { Write-Warning 'Icon generation failed; the default Python icon will be used.' }

# ---------------------------------------------------------------- 7. Launchers
Write-Step 'Creating launchers'

# A venv's pythonw.exe can be a redirector to the base interpreter, which makes
# Task Manager show "pythonw.exe" with the base install's path. This puts a
# real, renamed interpreter beside it and stamps the app icon into it.
& $venvPython (Join-Path $InstallDir 'tools\make_launcher.py') (Join-Path $venv 'Scripts') $iconPath
$launcherExe = Join-Path $venv 'Scripts\Myna.exe'
if (-not (Test-Path $launcherExe)) {
    Write-Warning 'Named launcher could not be created; falling back to pythonw.exe.'
    $launcherExe = $venvPythonw
}

$launcherCmd = Join-Path $InstallDir 'Myna.cmd'
@"
@echo off
rem Starts Myna with no console window.
start "" "%~dp0venv\Scripts\Myna.exe" -m app.main
"@ | Set-Content -Path $launcherCmd -Encoding ASCII

if (-not $NoShortcuts) {
    $startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
    New-Shortcut -LinkPath (Join-Path $startMenu "$AppName.lnk") `
                 -TargetPath $launcherExe -Arguments '-m app.main' `
                 -WorkingDirectory $InstallDir -IconPath $iconPath `
                 -Description 'Transcribe video and audio to subtitles and summaries, locally'
    Write-Note 'Start Menu shortcut created.'

    $desktop = [Environment]::GetFolderPath('Desktop')
    New-Shortcut -LinkPath (Join-Path $desktop "$AppName.lnk") `
                 -TargetPath $launcherExe -Arguments '-m app.main' `
                 -WorkingDirectory $InstallDir -IconPath $iconPath `
                 -Description 'Transcribe video and audio to subtitles and summaries, locally'
    Write-Note 'Desktop shortcut created.'
}

# ---------------------------------------------------------------- 8. Uninstall
Write-Step 'Registering with Add/Remove Programs'
$uninstallKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Myna'
New-Item -Path $uninstallKey -Force | Out-Null
$uninstallCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $InstallDir 'uninstall.ps1')`""
Set-ItemProperty -Path $uninstallKey -Name 'DisplayName'     -Value $AppName
Set-ItemProperty -Path $uninstallKey -Name 'DisplayVersion'  -Value '1.0.0'
Set-ItemProperty -Path $uninstallKey -Name 'Publisher'       -Value 'Local build'
Set-ItemProperty -Path $uninstallKey -Name 'InstallLocation' -Value $InstallDir
Set-ItemProperty -Path $uninstallKey -Name 'DisplayIcon'     -Value $iconPath
Set-ItemProperty -Path $uninstallKey -Name 'UninstallString' -Value $uninstallCommand
Set-ItemProperty -Path $uninstallKey -Name 'NoModify'        -Value 1 -Type DWord
Set-ItemProperty -Path $uninstallKey -Name 'NoRepair'        -Value 1 -Type DWord

# ---------------------------------------------------------------- 9. Storage
Write-Step 'Checking where models will be stored'
$storageReport = & $venvPython -c @"
import sys
sys.path.insert(0, r'$InstallDir')
from app.core.paths import models_root, set_models_root, free_space_gb

requested = r'''$ModelsDir'''.strip()
if requested:
    set_models_root(requested)

root = models_root()
free = free_space_gb(root)
print(f'Models folder: {root}')
print(f'Free space:    {free:.1f} GB')
if free < 8:
    print('WARNING: under 8 GB free. large-v3 (3.1 GB) plus a summary model')
    print('         (2 GB) will not fit. Use Model storage > Change... in the')
    print('         app, or re-run this installer with -ModelsDir D:\Models')
"@
Write-Note ($storageReport -join "`n    ")

# ---------------------------------------------------------------- 10. Report
Write-Step 'Checking GPU support'
$gpuReport = & $venvPython -c @"
import sys, os
sys.path.insert(0, r'$InstallDir')
from app.core.cuda_setup import ensure_cuda_dlls
ensure_cuda_dlls()
try:
    import ctranslate2
    count = ctranslate2.get_cuda_device_count()
except Exception as exc:
    count = 0
    print('note:', exc)
print('CUDA devices visible to Whisper:', count)
"@
Write-Note ($gpuReport -join "`n    ")

Write-Host ''
Write-Host '  Installed.' -ForegroundColor Green
Write-Host "  Launch it from the Start Menu or the Desktop shortcut named '$AppName'." -ForegroundColor White
Write-Host "  Location: $InstallDir" -ForegroundColor DarkGray
Write-Host '  Whisper models download automatically the first time you press Start.' -ForegroundColor DarkGray
Write-Host ''
