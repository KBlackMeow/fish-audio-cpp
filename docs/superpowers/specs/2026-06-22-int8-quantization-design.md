# INT8 Weight-Only Quantization (W8A16) Design

**Date:** 2026-06-22  
**Status:** approved  
**Target GPU:** RTX 4090 (Ada Lovelace, SM89, 24 GB VRAM)

## Motivation

Current FP16 inference loads ~8.7 GB weights into GPU memory (dual_ar ~8 GB + dac ~0.7 GB). WSL pinned-memory registration of the mmap'd model files is prohibitively slow, so weights must be uploaded eagerly. INT8 weight-only quantization halves the weight memory footprint to ~4.5 GB, leaving more room for KV cache and enabling larger batch sizes.

## Design Overview

Two phases, implemented in order:

### Phase 1 — Precision-suffix model naming

Rename model files with a `_<dtype>` suffix so multiple precision variants can coexist in the same directory. The server auto-detects the best available precision at startup.

- `dual_ar.bin` → `dual_ar_fp16.bin`
- `dac.bin` → `dac_fp16.bin`
- New: `dual_ar_int8.bin`, `dac_int8.bin` (produced by quantize tool)

**Auto-detection priority:** `int8` > `fp16` > `bf16` > unsuffixed (legacy).

### Phase 2 — INT8 W8A16 inference

Offline quantization with a Python script, runtime dequant+GEMM via a custom CUDA kernel.

#### File format

INT8 weights and their FP16 per-channel scales are stored in the existing `.bin` container:

- Weight tensor: `dtype=INT8` (new DType enum value `3`)
- Scale tensor: `dtype=FP16`, named `<original_name>_scale`, shape `[out_features]`

#### Quantization

- **Algorithm:** symmetric per-channel (row-wise for GEMM weights)
  - `scale[i] = max(abs(W[i,:])) / 127.0`
  - `W_int8[i,j] = round(W_fp16[i,j] / scale[i])` clamped to `[-127, 127]`
- **What is quantized:** all Linear/GEMM weight matrices (wqkv, wo, w_gate, w_up, w_down, embeddings)
- **What stays FP16:** RMSNorm weights, biases, RoPE frequencies, DAC codebooks (small <0.1%)

#### Runtime GEMM

Custom CUDA kernel `int8_dequant_gemm_fp16`:
- Input: INT8 weight [M,K] + FP16 scale [M] + FP16 activation [N,K]
- Output: FP16 result [N,M]
- Inner loop: `accum += (int)W_int8[m,k] * scale[m] * (float)X[n,k]`
- 128-thread tiled design with shared-memory caching of the INT8 weight tile

#### Accuracy validation

Same input (text + seed), compare fp16 vs int8 WAV output. Expected: negligible audible difference; SNR > 40 dB between output waveforms.

## Files Changed / Added

| File | Action | Purpose |
|------|--------|---------|
| `tools/quantize_int8.py` | **New** | Read FP16 .bin, write INT8 .bin + FP16 scales |
| `src/kernels/int8_gemm.cu` | **New** | INT8 dequant + FP16 GEMM kernel |
| `src/kernels/kernels.h` | Modify | Declare int8_gemm entry point |
| `src/model/tensor.h` | Modify | Add `INT8 = 3` to DType enum |
| `src/model/loader.cc` | Modify | Handle INT8 dtype in load path |
| `src/engine/dual_ar_engine.cc` | Modify | INT8 weight upload + quantized GEMM dispatch |
| `src/engine/dac_engine.cc` | Modify | INT8 path for out_proj / dense weights |
| `src/main.cc` | Modify | Precision-suffix auto-detection |
| `CMakeLists.txt` | Modify | Add `int8_gemm.cu` to fish_kernels |
| `scripts/test_en.sh` | Modify | Update model paths |
| `scripts/build_and_test.sh` | Modify | Update model paths |

## Memory Budget (estimated)

| Component | FP16 | INT8 |
|-----------|------|------|
| dual_ar weights | 8.0 GB | 4.0 GB + 0.016 GB scales |
| dac weights | 0.7 GB | 0.35 GB + negligible scales |
| KV cache (512 tokens) | ~0.3 GB | unchanged |
| Workspace buffers | ~0.1 GB | unchanged |
| **Total** | **~9.1 GB** | **~4.8 GB** |
