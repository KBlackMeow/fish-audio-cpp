# Windows build script for fish-audio-cpp
# Requirements:
#   - CUDA Toolkit 12.4+ installed (https://developer.nvidia.com/cuda-downloads)
#   - cuDNN 9.x installed (https://developer.nvidia.com/cudnn)
#   - Visual Studio 2022 with "Desktop development with C++" workload
#   - CMake 3.24+ (already installed)
#   - Git (for FetchContent dependencies)
#
# Usage:
#   .\scripts\build_windows.ps1
#   .\scripts\build_windows.ps1 -BuildType Debug
#   .\scripts\build_windows.ps1 -Jobs 8

param(
    [string]$BuildType = "Release",
    [int]$Jobs = $env:NUMBER_OF_PROCESSORS,
    [string]$BuildDir = ""
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
if (-not $BuildDir) { $BuildDir = Join-Path $RootDir "build_win" }

Write-Host "== fish-audio-cpp Windows Build ==" -ForegroundColor Cyan
Write-Host "  Root   : $RootDir"
Write-Host "  Build  : $BuildDir"
Write-Host "  Config : $BuildType"
Write-Host "  Jobs   : $Jobs"

# ── Verify CUDA ──────────────────────────────────────────────────────────────
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath -or -not (Test-Path $cudaPath)) {
    # Try common CUDA install locations
    $candidates = @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $cudaPath = $c; break }
    }
}
if (-not $cudaPath -or -not (Test-Path $cudaPath)) {
    Write-Host ""
    Write-Host "ERROR: CUDA Toolkit not found." -ForegroundColor Red
    Write-Host "Please install CUDA Toolkit 12.4+ from:"
    Write-Host "  https://developer.nvidia.com/cuda-downloads"
    Write-Host "Then re-run this script."
    exit 1
}
Write-Host "  CUDA   : $cudaPath" -ForegroundColor Green

$nvcc = Join-Path $cudaPath "bin\nvcc.exe"
if (-not (Test-Path $nvcc)) {
    Write-Host "ERROR: nvcc not found at $nvcc" -ForegroundColor Red
    exit 1
}

# Add CUDA bin to PATH for this session
$env:PATH = "$cudaPath\bin;$env:PATH"

# ── Verify cuDNN ─────────────────────────────────────────────────────────────
$cudnnHeader = $null
$cudnnSearchPaths = @(
    "$cudaPath\include\cudnn.h",
    "$env:CUDNN_PATH\include\cudnn.h",
    "C:\Program Files\NVIDIA\CUDNN\v9.x\include\cudnn.h"
)
foreach ($p in $cudnnSearchPaths) {
    if (Test-Path $p) { $cudnnHeader = $p; break }
}
if (-not $cudnnHeader) {
    Write-Host ""
    Write-Host "WARNING: cudnn.h not found. CMake configure will fail." -ForegroundColor Yellow
    Write-Host "Please install cuDNN 9.x from:"
    Write-Host "  https://developer.nvidia.com/cudnn"
    Write-Host "Then set the CUDNN_PATH environment variable to the cuDNN install directory."
}

# ── Configure ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "== Configure ==" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$cmakeArgs = @(
    "-S", $RootDir,
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DCMAKE_CUDA_ARCHITECTURES=89"  # RTX 4090; change for your GPU
)
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configure FAILED." -ForegroundColor Red
    exit $LASTEXITCODE
}

# ── Build ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "== Build ==" -ForegroundColor Cyan
& cmake --build $BuildDir --config $BuildType --target fish-server -- /maxcpucount:$Jobs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build FAILED." -ForegroundColor Red
    exit $LASTEXITCODE
}

& cmake --build $BuildDir --config $BuildType --target test_fish -- /maxcpucount:$Jobs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Test build FAILED (non-fatal)." -ForegroundColor Yellow
}

# ── Locate outputs ───────────────────────────────────────────────────────────
$serverExe = Get-ChildItem -Recurse -Filter "fish-server.exe" -Path $BuildDir | Select-Object -First 1
$testExe   = Get-ChildItem -Recurse -Filter "test_fish.exe"   -Path $BuildDir | Select-Object -First 1

Write-Host ""
Write-Host "== Done ==" -ForegroundColor Green
if ($serverExe) { Write-Host "  fish-server : $($serverExe.FullName)" }
if ($testExe)   { Write-Host "  test_fish   : $($testExe.FullName)" }

Write-Host ""
Write-Host "To run tests:"
if ($testExe) { Write-Host "  & `"$($testExe.FullName)`"" }
Write-Host ""
Write-Host "To run inference:"
if ($serverExe) {
    Write-Host "  & `"$($serverExe.FullName)`" --model-dir checkpoints/s2-pro --text `"Hello world`" --output output/test.wav"
}
