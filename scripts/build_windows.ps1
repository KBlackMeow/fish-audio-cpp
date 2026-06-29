param(
    [string]$BuildType = "Release",
    [string]$BuildDir  = "",
    [string]$CudaArch  = "89"
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
if (-not $BuildDir) { $BuildDir = Join-Path $RootDir "build_win" }

# Locate CUDA
$CudaPath = $env:CUDA_PATH
if (-not $CudaPath -or -not (Test-Path $CudaPath)) {
    foreach ($c in @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    )) {
        if (Test-Path $c) { $CudaPath = $c; break }
    }
}
if (-not $CudaPath) {
    Write-Host "ERROR: CUDA Toolkit not found. Install from https://developer.nvidia.com/cuda-downloads" -ForegroundColor Red
    exit 1
}
Write-Host "CUDA : $CudaPath" -ForegroundColor Cyan
$env:PATH = "$CudaPath\bin;$env:PATH"

# Configure
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
& cmake -S $RootDir -B $BuildDir `
    -DCMAKE_BUILD_TYPE=$BuildType `
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch" `
    "-T" "v143,cuda=$CudaPath" `
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

# Build
& cmake --build $BuildDir --config $BuildType --target fish-server -j
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

Write-Host ""
Write-Host "Done: $BuildDir\$BuildType\fish-server.exe" -ForegroundColor Green
