# INT8 Weight-Only Quantization (W8A16) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add INT8 weight-only quantization (W8A16) inference support, reducing GPU memory ~48% while maintaining audio quality.

**Architecture:** Two-phase approach. Phase 1 renames model files with precision suffixes and adds auto-detection in `main.cc`. Phase 2 implements offline INT8 quantization via Python script, a custom CUDA kernel for dequant+GEMM, and engine integration for both DualAR and DAC engines.

**Tech Stack:** CUDA 12.4, cuBLAS, C++17, Python 3 (numpy), CMake 3.24

## Global Constraints

- Target GPU: RTX 4090 (SM89, Ada Lovelace)
- CUDA architecture: `89`
- C++ standard: C++17
- INT8 quantization: symmetric per-channel, scale stored as FP16
- Auto-detection priority: int8 > fp16 > bf16 > unsuffixed (legacy)
- RMSNorm/biases/codebooks remain FP16 (not quantized)
- Must pass existing test suite, build with `scripts/build_and_test.sh`

---

## File Structure

```
fish-audio-cpp/
├── tools/
│   └── quantize_int8.py            ← NEW
├── src/
│   ├── kernels/
│   │   ├── int8_gemm.cu            ← NEW
│   │   └── kernels.h               ← MODIFY
│   ├── model/
│   │   └── tensor.h                ← MODIFY
│   ├── engine/
│   │   ├── dual_ar_engine.cc       ← MODIFY
│   │   ├── dual_ar_engine.h        ← MODIFY
│   │   ├── dac_engine.cc           ← MODIFY
│   │   └── dac_engine.h            ← MODIFY
│   └── main.cc                     ← MODIFY
├── CMakeLists.txt                  ← MODIFY
└── scripts/*.sh                    ← MODIFY (6 files)
```

## Key Interfaces

```cpp
// tensor.h
enum class DType : uint32_t { FP32=0, FP16=1, BF16=2, INT8=3 };

// kernels.h
void int8_dequant_gemm_fp16(const int8_t* W, const __half* scale,
                            const __half* X, __half* Y,
                            int M, int N, int K, cudaStream_t s=0);

// main.cc
static std::string resolve_model_path(const std::string& dir,
                                       const std::string& basename);

// dual_ar_engine.h (new private members)
bool use_int8_ = false;
std::unordered_map<void*, __half*> weight_to_scale_;
void quantized_gemm(int M, int N, int K, const TensorView& w,
                    const __half* X, __half* Y);
```

---

### Task 1: Rename model files + update scripts

- [ ] **Step 1: Rename model files**

```bash
cd /home/illya/fish-audio-cpp/checkpoints/s2-pro
mv dual_ar.bin dual_ar_fp16.bin
mv dac.bin dac_fp16.bin
ls -lh dual_ar_fp16.bin dac_fp16.bin
```

- [ ] **Step 2: Update all 6 shell scripts (lines 28-29 each)**

Replace `dual_ar.bin` → `dual_ar_fp16.bin`, `dac.bin` → `dac_fp16.bin` in:
`scripts/test_en.sh`, `test_zh.sh`, `test_ja.sh`, `test_mix.sh`, `test_mix_000047.sh`, `build_and_test.sh`

- [ ] **Step 3: Commit**

```bash
git add checkpoints/ scripts/
git commit -m "refactor: rename model files with _fp16 precision suffix"
```

---

### Task 2: DType::INT8 + precision auto-detection

- [ ] **Step 1: tensor.h — add INT8 to DType**

```cpp
enum class DType : uint32_t { FP32 = 0, FP16 = 1, BF16 = 2, INT8 = 3 };
// dtype_size(DType::INT8) returns 1
```

- [ ] **Step 2: main.cc — add resolve_model_path() helper (after write_wav)**

```cpp
static std::string resolve_model_path(const std::string& dir,
                                       const std::string& base) {
    static const char* sfx[] = {"_int8", "_fp16", "_bf16", ""};
    for (auto s : sfx) {
        auto p = dir + "/" + base + s + ".bin";
        if (std::filesystem::exists(p)) { spdlog::info("Resolved: {}", p); return p; }
    }
    return dir + "/" + base + "_fp16.bin";
}
```

- [ ] **Step 3: main.cc — replace hardcoded paths**

```cpp
// OLD: model_dir + "/dual_ar.bin"
// NEW: resolve_model_path(model_dir, "dual_ar")
// Same for dac.
```

- [ ] **Step 4: Build & smoke test**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
./build/fish-server --model-dir checkpoints/s2-pro --text "Hello" --output /tmp/t2.wav --max-tokens 16
```

Expected: log shows "Resolved: .../dual_ar_fp16.bin", generates audio.

- [ ] **Step 5: Commit**

```bash
git add src/model/tensor.h src/main.cc
git commit -m "feat: DType::INT8 + precision-suffix model auto-detection"
```

---

### Task 3: quantize_int8.py

- [ ] **Step 1: Create tools/quantize_int8.py**

Full script at: `tools/quantize_int8.py`

Key behavior:
- Reads FISH .bin format (magic 0x46495348)
- Symmetric per-channel: `scale[i] = max(|W[i,:]|)/127.0`, `W_int8[i,j] = round(W[i,j]/scale[i])` clamped to [-127,127]
- Skips patterns: `_norm.`, `.bias`, `codebook.weight`, `codebook_embeddings`, `freqs_cis`, `alpha`, `gamma`, `out_proj.bias`, `in_proj.bias`
- Output: INT8 weight tensor + `_scale` FP16 tensor per quantized weight
- Same .bin container format with 256-byte alignment

- [ ] **Step 2: Run quantize on both models**

```bash
python tools/quantize_int8.py checkpoints/s2-pro/dual_ar_fp16.bin checkpoints/s2-pro/dual_ar_int8.bin
python tools/quantize_int8.py checkpoints/s2-pro/dac_fp16.bin checkpoints/s2-pro/dac_int8.bin
```

- [ ] **Step 3: Verify output**

```bash
python -c "
import struct, os
for p in ['dual_ar_int8.bin','dac_int8.bin']:
    with open(f'checkpoints/s2-pro/{p}','rb') as f:
        m=struct.unpack('<I',f.read(4))[0]; v=struct.unpack('<I',f.read(4))[0]; n=struct.unpack('<I',f.read(4))[0]
        print(f'{p}: magic=0x{m:08X} ver={v} tensors={n} size={os.path.getsize(f\"checkpoints/s2-pro/{p}\")/1e6:.0f}MB')"
```

- [ ] **Step 4: Commit**

```bash
git add tools/quantize_int8.py
git commit -m "feat: add INT8 offline quantization script (Python)"
```

---

### Task 4: INT8 dequant+GEMM CUDA kernel

- [ ] **Step 1: Create src/kernels/int8_gemm.cu**

```cuda
// Y = (W_int8 * scale) × X^T
// Grid: (M, N), Block: 256. Each block = one output element.
// K-tiled: shared memory cooperative X load, warp reduce.

#include "kernels/kernels.h"
#include <cuda_fp16.h>
#include <cstdint>

namespace fish::kernels {
namespace {
static constexpr int TILE_K = 128, BLOCK = 256;

__global__ void int8_gemm_k(const int8_t* __restrict__ W,
                            const __half* __restrict__ scale,
                            const __half* __restrict__ X,
                            __half* __restrict__ Y,
                            int M, int N, int K) {
    int r = blockIdx.x, c = blockIdx.y;
    if (r >= M || c >= N) return;
    __shared__ __half xs[TILE_K];
    float s = __half2float(scale[r]);
    auto* wr = W + (size_t)r * K;
    auto* xr = X + (size_t)c * K;
    float acc = 0;
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        int lim = min(TILE_K, K - k0);
        for (int i = threadIdx.x; i < lim; i += BLOCK) xs[i] = xr[k0+i];
        __syncthreads();
        int each = (lim + BLOCK - 1) / BLOCK;
        for (int ki = threadIdx.x*each, end=min(ki+each,lim); ki<end; ki++)
            acc += (float)wr[k0+ki] * s * __half2float(xs[ki]);
        __syncthreads();
    }
    for (int o=16; o; o>>=1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (!(threadIdx.x & 31)) Y[r + (size_t)c * M] = __float2half(acc);
}
} // anon

void int8_dequant_gemm_fp16(const int8_t* W, const __half* scale,
                             const __half* X, __half* Y,
                             int M, int N, int K, cudaStream_t s) {
    int8_gemm_k<<<dim3(M,N), dim3(BLOCK), 0, s>>>(W, scale, X, Y, M, N, K);
}
} // fish::kernels
```

- [ ] **Step 2: Declare in kernels.h** (inside `namespace fish::kernels`)

```cpp
void int8_dequant_gemm_fp16(const int8_t* W, const __half* scale,
                             const __half* X, __half* Y,
                             int M, int N, int K, cudaStream_t s = 0);
```

- [ ] **Step 3: Add to CMakeLists.txt** — `src/kernels/int8_gemm.cu` in fish_kernels

- [ ] **Step 4: Build kernels**

```bash
cmake --build build --target fish_kernels -j$(nproc)
```

- [ ] **Step 5: Commit**

```bash
git add src/kernels/int8_gemm.cu src/kernels/kernels.h CMakeLists.txt
git commit -m "feat: INT8 dequant+GEMM CUDA kernel"
```

---

### Task 5: DualAREngine INT8 path

- [ ] **Step 1: dual_ar_engine.h — add members**

```cpp
// In private section:
#include <unordered_map>
bool use_int8_ = false;
std::unordered_map<void*, __half*> weight_to_scale_;
void quantized_gemm(int M, int N, int K, const TensorView& w,
                    const __half* X, __half* Y);
```

- [ ] **Step 2: dual_ar_engine.cc — rewrite init() weight upload**

Replace the flat `wv` loop with `NamedTV` approach: each weight paired with
its .bin key name so we can look up `<name>_scale` for INT8 tensors.
Upload weight → GPU, then check `loader_.has(name + "_scale")` and upload
scale if present. Store `weight_to_scale_[gpu_ptr] = scale_gpu_ptr`.

- [ ] **Step 3: Add quantized_gemm() method**

```cpp
void DualAREngine::quantized_gemm(int M_out, int N_out, int K,
                                   const TensorView& w,
                                   const __half* X, __half* Y) {
    auto it = weight_to_scale_.find(w.data);
    if (it != weight_to_scale_.end() && it->second) {
        kernels::int8_dequant_gemm_fp16(
            static_cast<const int8_t*>(w.data), it->second,
            X, Y, M_out, N_out, K, stream_);
    } else {
        gemm_fp16(M_out, N_out, K, w.as<__half>(), X, Y, cublas_);
    }
}
```

- [ ] **Step 4: Replace all gemm_fp16 calls** in prefill/decode_step/get_logits/fast_codebook_decode with `quantized_gemm`. (~20 call sites, change `lw.xxx.as<__half>()` → `lw.xxx`, drop `cublas_` param)

- [ ] **Step 5: Commit**

```bash
git add src/engine/dual_ar_engine.h src/engine/dual_ar_engine.cc
git commit -m "feat: DualAREngine INT8 weight path"
```

---

### Task 6: DACEngine INT8 path

- [ ] **Step 1: dac_engine.h — add DenseWeight.scale + members**

```cpp
struct DenseWeight {
    __half* weight=nullptr; __half* bias=nullptr;
    __half* scale=nullptr;  // NEW: INT8 per-channel scale
    int out=0, in=0;
};
// private:
bool use_int8_ = false;
std::unordered_map<void*, __half*> weight_to_scale_;
```

- [ ] **Step 2: dac_engine.cc — quantized_gemm_fp16 helper**

```cpp
static void qgemm(int M, int N, int K, const __half* W, const __half* S,
                  const __half* X, __half* Y, cublasHandle_t cb, cudaStream_t st) {
    if (S) kernels::int8_dequant_gemm_fp16((const int8_t*)W, S, X, Y, M, N, K, st);
    else gemm_fp16(M, N, K, W, X, Y, cb);
}
```

- [ ] **Step 3: load_matrix/load_dense — detect & upload INT8 scales**

- [ ] **Step 4: Replace gemm_fp16 calls in dense_cf/dense_nobias/post_module/enc_transformer** with `qgemm(..., dw.scale, ...)`

- [ ] **Step 5: Commit**

```bash
git add src/engine/dac_engine.h src/engine/dac_engine.cc
git commit -m "feat: DACEngine INT8 weight path"
```

---

### Task 7: Build, test, validate

- [ ] **Step 1: Full clean build**

```bash
rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

- [ ] **Step 2: Run unit tests**

```bash
./build/test_fish
```

- [ ] **Step 3: INT8 smoke test**

```bash
./build/fish-server --model-dir checkpoints/s2-pro --text "Hello world" \
  --output /tmp/int8_test.wav --max-tokens 32 --seed 42
```

Expected: log shows "Resolved: .../dual_ar_int8.bin" and "GPU weights: XXXX MB (INT8)". Audio generated.

- [ ] **Step 4: FP16 vs INT8 quality comparison**

```bash
# Generate both with same seed, same text (longer utterance)
# Compute SNR between outputs
python -c "
import numpy as np, wave, struct
def rms(a,b):
    s=np.mean(a.astype(float)**2)
    n=np.mean((a.astype(float)-b.astype(float))**2)
    return 10*np.log10(s/max(n,1e-10))
# ... (read both WAVs, compute SNR)
"
```

Expected: SNR > 40 dB.

- [ ] **Step 5: Final commit**

```bash
git commit -m "test: validate INT8 quantization end-to-end"
```
