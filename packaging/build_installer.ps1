# Build Virtual Steering Wheel for Windows 10/11 x64.
# Run this script from a Windows machine with Python 3.10 or 3.11 (64-bit).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ---- Verify interpreter ----
Write-Host "Checking Python..." -ForegroundColor Cyan
$ver = python -c "import sys,struct; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor) + ' ' + str(struct.calcsize('P')*8) + 'bit')"
if (!($ver -match "^3\.(10|11|12) 64bit$")) {
    Write-Error "Python 3.10-3.12 64-bit is required.  Found: $ver"
}

# ---- Isolated virtual environment (reproducible) ----
Write-Host "Creating clean virtual environment..." -ForegroundColor Cyan
if (Test-Path ".venv") { Remove-Item -Recurse -Force ".venv" }
python -m venv .venv
.\.venv\Scripts\Activate.ps1

Write-Host "Installing dependencies into venv..." -ForegroundColor Cyan
pip install --upgrade pip
pip install -r ..\requirements.txt pyinstaller

# ---- Build the one-file EXE ----
Write-Host "Building bundled application..." -ForegroundColor Cyan
pyinstaller --clean --noconfirm virtual_steering_wheel.spec

# ---- Verify output ----
$exePath = Join-Path $PSScriptRoot "..\dist\VirtualSteeringWheel.exe"
if (!(Test-Path $exePath)) {
    Write-Error "PyInstaller did not produce $exePath. Check the log above."
}
$size = (Get-Item $exePath).Length / 1MB
Write-Host "Created $exePath ($([math]::Round($size,1)) MB)" -ForegroundColor Green

# ---- Find Inno Setup ----
$iscc = Get-Command iscc.exe -ErrorAction SilentlyContinue
if (-not $iscc) {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { $iscc = @{ Source = $candidate }; break }
    }
}

if (-not $iscc) {
    Write-Error "Inno Setup 6 was not found. Install it from https://jrsoftware.org/isinfo.php, then run this script again."
}

# ---- Build setup wizard + uninstaller ----
Write-Host "Creating setup wizard and uninstaller..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path installer | Out-Null
& $iscc.Source installer.iss

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Green
Write-Host "  BUILD COMPLETE" -ForegroundColor Green
Write-Host "  Installer: $PSScriptRoot\installer\VirtualSteeringWheel-Setup.exe" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
