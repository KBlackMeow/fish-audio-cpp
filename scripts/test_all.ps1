# scripts\test_all.ps1 — build + full test matrix across all 3 models
#
# Models:
#   s2-pro-fp16
#   s2-pro-int8-w8a8-g256
#   s2-pro-int8-w8a8-g64
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\test_all.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\test_all.ps1 -NoBuild

[CmdletBinding()]
param(
    [switch]$NoBuild,
    [string]$BuildDir  = "",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

# 切换控制台到 UTF-8，确保中文参数正确传给 exe
chcp 65001 | Out-Null
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$RootDir   = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not $BuildDir)  { $BuildDir  = Join-Path $RootDir "build_win" }
if (-not $OutputDir) { $OutputDir = Join-Path $RootDir "output" }

$Bin  = Join-Path $BuildDir "Release\fish-server.exe"
$Seed = 42

$CudaPath   = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
$CudaArch   = "89"   # RTX 4090 — change if needed (RTX 3090/3080 → 86, RTX 2080 → 75)
$Toolset    = "v143,cuda=$CudaPath"

# 把 CUDA / cuDNN DLL 目录加入当前会话 PATH，让 exe 能找到运行时库
$env:PATH = "$CudaPath\bin;C:\Program Files\NVIDIA\CUDNN\v9.23\bin\12.9\x64;$env:PATH"

# ── models ────────────────────────────────────────────────────────────────────

$Models = @(
    @{ Tag = "fp16";      Dir = Join-Path $RootDir "models\s2-pro-fp16" },
    @{ Tag = "int8-g256"; Dir = Join-Path $RootDir "models\s2-pro-int8-w8a8-g256" },
    @{ Tag = "int8-g64";  Dir = Join-Path $RootDir "models\s2-pro-int8-w8a8-g64" }
)

# ── helpers ────────────────────────────────────────────────────────────────────

function Check-File([string]$Path) {
    if (-not (Test-Path $Path)) {
        Write-Host "MISSING: $Path" -ForegroundColor Red
        exit 1
    }
}

function Run-One {
    param(
        [string]$Tag,
        [string]$ModelDir,
        [string[]]$ExtraArgs
    )
    $OutWav = Join-Path $OutputDir "$Tag.wav"
    Write-Host "  -> $Tag" -ForegroundColor Cyan
    $allArgs = @(
        "--model-dir", $ModelDir,
        "--output", $OutWav,
        "--seed", $Seed
    ) + $ExtraArgs

    & $Bin @allArgs 2>&1 | Where-Object {
        $_ -match "Resolved|first sem|Generated|Audio|WAV|GPU"
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED (exit $LASTEXITCODE): $Tag" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# ── build ──────────────────────────────────────────────────────────────────────

if (-not $NoBuild) {
    Write-Host "========== BUILD ==========" -ForegroundColor Yellow
    Check-File (Join-Path $RootDir "CMakeLists.txt")
    foreach ($m in $Models) {
        Check-File (Join-Path $m.Dir "dual_ar.bin")
        Check-File (Join-Path $m.Dir "dac.bin")
    }

    & cmake -S $RootDir -B $BuildDir `
        -DCMAKE_BUILD_TYPE=Release `
        "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch" `
        "-T" $Toolset `
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

    & cmake --build $BuildDir --config Release --target fish-server -j
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }
    Write-Host ""
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# ── unit tests ─────────────────────────────────────────────────────────────────

Write-Host "========== UNIT TESTS ==========" -ForegroundColor Yellow
& cmake --build $BuildDir --config Release --target test_fish -j 2>&1 | Out-Null
$TestBin = Join-Path $BuildDir "Release\test_fish.exe"
if (Test-Path $TestBin) {
    & $TestBin 2>&1 | Select-Object -Last 2
} else {
    Write-Host "  test_fish.exe not found, skipping" -ForegroundColor DarkYellow
}
Write-Host ""

# ── iterate over models ────────────────────────────────────────────────────────

foreach ($m in $Models) {
    Write-Host "========== MODEL: $($m.Tag) ==========" -ForegroundColor Yellow

    # ── pure text, English ──────────────────────────────────────────────────
    Write-Host "--- pure text (English) ---"
    Run-One -Tag "$($m.Tag)_english" -ModelDir $m.Dir -ExtraArgs @(
        "--text", "Hello world, this is a test of the speech synthesis system.",
        "--max-tokens", "128"
    )

    # ── pure text, Chinese ──────────────────────────────────────────────────
    Write-Host "--- pure text (Chinese) ---"
    Run-One -Tag "$($m.Tag)_chinese" -ModelDir $m.Dir -ExtraArgs @(
        "--text", "你好世界，这是一个语音合成测试。今天天气真好，我们出去散步吧。",
        "--max-tokens", "128"
    )

    # ── voice clone, Nahida (English) ───────────────────────────────────────
    Write-Host "--- voice clone: Nahida ---"
    Check-File (Join-Path $RootDir "example\vo_LLZAQ001_4_nahida_03.wav")
    Check-File (Join-Path $RootDir "example\vo_LLZAQ001_4_nahida_03.lab")
    Run-One -Tag "$($m.Tag)_nahida" -ModelDir $m.Dir -ExtraArgs @(
        "--ref-audio", (Join-Path $RootDir "example\vo_LLZAQ001_4_nahida_03.wav"),
        "--ref-text",  (Join-Path $RootDir "example\vo_LLZAQ001_4_nahida_03.lab"),
        "--text", "Traveler, shall we go on an adventure today? Whether it's the winds of Mondstadt or the mountains of Liyue, I'll be right by your side.",
        "--max-tokens", "256"
    )

    # ── voice clone, 000047 (Chinese) ───────────────────────────────────────
    Write-Host "--- voice clone: 000047 ---"
    Check-File (Join-Path $RootDir "example\000047.wav")
    Check-File (Join-Path $RootDir "example\000047.lab")
    Run-One -Tag "$($m.Tag)_000047" -ModelDir $m.Dir -ExtraArgs @(
        "--ref-audio", (Join-Path $RootDir "example\000047.wav"),
        "--ref-text",  (Join-Path $RootDir "example\000047.lab"),
        "--text", "今天天气真好，我们出去散步吧。阳光明媚，微风拂面，让人心情愉悦。",
        "--max-tokens", "256"
    )

    Write-Host ""
}

# ── summary ────────────────────────────────────────────────────────────────────

Write-Host "========== ALL OUTPUTS ==========" -ForegroundColor Yellow
Get-ChildItem (Join-Path $OutputDir "*.wav") |
    Sort-Object Length -Descending |
    Format-Table Name, @{N="Size(KB)"; E={[math]::Round($_.Length/1KB,1)}}, LastWriteTime -AutoSize
Write-Host ""
Write-Host "Done -> $OutputDir" -ForegroundColor Green
