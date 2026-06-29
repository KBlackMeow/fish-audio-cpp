# fish-audio-cpp

纯 C++/CUDA 实现的 Fish Audio S2 Pro TTS 推理引擎。支持 CLI 和 HTTP Server，支持参考音频（voice cloning）。

**支持平台：Linux / Windows（CUDA 12.4+）**

## 目录

- [模型准备](#模型准备)
- [编译](#编译)
  - [Linux](#linux)
  - [Windows](#windows)
- [CLI 使用](#cli-使用)
- [HTTP Server](#http-server)
- [API 接口](#api-接口)
- [Python 测试脚本](#python-测试脚本)

---

## 模型准备

### 1. 下载原版模型

```bash
# 从 HuggingFace 下载（约 11 GB）
pip install huggingface_hub
huggingface-cli download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

原始模型文件结构：

```
checkpoints/s2-pro/
├── config.json                      # 模型架构配置
├── model-00001-of-00002.safetensors # ~5 GB
├── model-00002-of-00002.safetensors # ~4 GB
├── model.safetensors.index.json
├── codec.pth                        # DAC codec 权重 (~1.9 GB)
├── tokenizer.json                   # BPE tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
└── chat_template.jinja
```

### 2. 转换为 FP16 格式

原版模型是 BF16 safetensors，需要转换为本项目使用的 `.bin` 格式：

```bash
# 转换 DualAR（文本→VQ codes 的大模型）
python3 tools/step1_convert_fp16.py \
  --model dual_ar \
  --checkpoint checkpoints/s2-pro \
  --output-dir checkpoints/s2-pro

# 转换 DAC（VQ codes→音频的 codec）
python3 tools/step1_convert_fp16.py \
  --model dac \
  --checkpoint checkpoints/s2-pro/codec.pth \
  --output-dir checkpoints/s2-pro
```

生成文件：

```
checkpoints/s2-pro/
├── dual_ar.bin       # ~9 GB（FP16）
├── dual_ar_config.json
├── dac.bin           # ~1.5 GB（FP16）
├── dac_config.json
└── ...（原始文件不变）
```

### 3. 模型目录布局

推理时使用 `models/<model>-<dtype>/` 目录，通过符号链接指向实际文件：

**Linux：**

```bash
mkdir -p models/s2-pro-fp16
cd models/s2-pro-fp16

ln -s ../../checkpoints/s2-pro/dual_ar.bin dual_ar.bin
ln -s ../../checkpoints/s2-pro/dual_ar_config.json dual_ar_config.json
ln -s ../../checkpoints/s2-pro/dac.bin dac.bin
ln -s ../../checkpoints/s2-pro/dac_config.json dac_config.json
ln -s ../../checkpoints/s2-pro/tokenizer.json tokenizer.json
ln -s ../../checkpoints/s2-pro/tokenizer_config.json tokenizer_config.json
ln -s ../../checkpoints/s2-pro/special_tokens_map.json special_tokens_map.json
```

**Windows（PowerShell，需开发者模式或管理员权限）：**

```powershell
New-Item -ItemType Directory -Force models\s2-pro-fp16
cd models\s2-pro-fp16
cmd /c mklink dual_ar.bin ..\..\checkpoints\s2-pro\dual_ar.bin
cmd /c mklink dual_ar_config.json ..\..\checkpoints\s2-pro\dual_ar_config.json
cmd /c mklink dac.bin ..\..\checkpoints\s2-pro\dac.bin
cmd /c mklink dac_config.json ..\..\checkpoints\s2-pro\dac_config.json
cmd /c mklink tokenizer.json ..\..\checkpoints\s2-pro\tokenizer.json
cmd /c mklink tokenizer_config.json ..\..\checkpoints\s2-pro\tokenizer_config.json
cmd /c mklink special_tokens_map.json ..\..\checkpoints\s2-pro\special_tokens_map.json
```

> **提示**：也可以直接把文件复制进去（无需权限），`mklink` 只是节省磁盘空间。

目录结构：

```
models/
├── s2-pro-fp16/
│   ├── dual_ar.bin -> ../../checkpoints/s2-pro/dual_ar.bin
│   ├── dual_ar_config.json -> ...
│   ├── dac.bin -> ...
│   ├── dac_config.json -> ...
│   ├── tokenizer.json -> ...
│   ├── tokenizer_config.json -> ...
│   └── special_tokens_map.json -> ...
├── s2-pro-int8-w8a8-g256/   # INT8 模型（同样布局）
└── s2-pro-int8-w8a8-g64/    # INT8 模型（更小 group size）
```

### 4. INT8 量化（可选，显存需求减半）

如果 GPU 显存不足（如 ≤16GB），建议做 INT8 量化。

#### 4.1 采集校准数据

```bash
# 启动 FP16 服务器，采集各层的激活值分布
FISH_CALIBRATE_DIR=/tmp/calib ./build/fish-server \
  --model-dir models/s2-pro-fp16 \
  --server --port 8080

# 另开终端，发送多样化文本进行校准
python3 tools/collect_calibration_w8a8.py \
  --host 127.0.0.1 --port 8080 \
  --output-dir /tmp/calib
```

#### 4.2 导出 INT8 模型

```bash
# 导出 dual_ar（W8A8 + AWQ + GPTQ）
python3 tools/step2_export_w8a8.py \
  /tmp/calib \
  checkpoints/s2-pro/dual_ar.bin \
  checkpoints/s2-pro/dual_ar_int8-w8a8.bin \
  --device cuda:0

# 创建模型目录
mkdir -p models/s2-pro-int8-w8a8-g256
cd models/s2-pro-int8-w8a8-g256
ln -s ../../checkpoints/s2-pro/dual_ar_int8-w8a8.bin dual_ar.bin
ln -s ../../checkpoints/s2-pro/dac_fp16.bin dac.bin          # DAC 保持 FP16
# ... 其他 config 和 tokenizer 同理
```

> **注意**：DAC 模型不支持 INT8，始终使用 FP16。但 DAC 只有 ~1.5 GB，对显存影响不大。

---

## 编译

其余依赖（spdlog, httplib, nlohmann/json, sentencepiece, googletest）通过 CMake FetchContent 自动下载，无需手动安装。

### Linux

**依赖：**

- CUDA Toolkit ≥ 12.4
- cuDNN
- CMake ≥ 3.24
- GCC ≥ 9（C++17）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target fish-server -j$(nproc)

# 可选：单元测试
cmake --build build --target test_fish -j$(nproc)
./build/test_fish

# 全模型测试矩阵（3 种精度 × 4 种场景）
bash scripts/test_all.sh
```

### Windows

**依赖：**

- [CUDA Toolkit 12.4+](https://developer.nvidia.com/cuda-downloads)
- [cuDNN 9.x](https://developer.nvidia.com/cudnn)（解压后放到 `C:\Program Files\NVIDIA\CUDNN\v9.x\`）
- [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)（选"Desktop development with C++"工作负载）
- CMake 3.24+
- PowerShell 5.1+（Windows 内置）

> **注意**：CUDA 12.9 不支持 VS 2026（MSVC 19.51+），必须使用 VS 2022（MSVC 19.4x）。

**一键构建：**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
```

**手动构建：**

```powershell
$CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
cmake -S . -B build_win -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_CUDA_ARCHITECTURES=89 `
      "-T" "v143,cuda=$CudaPath" `
      "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
cmake --build build_win --config Release --target fish-server -j
```

可执行文件位于 `build_win\Release\fish-server.exe`。

**全模型测试：**

```powershell
# 构建 + 测试
powershell -ExecutionPolicy Bypass -File scripts\test_all.ps1

# 仅测试（跳过编译）
powershell -ExecutionPolicy Bypass -File scripts\test_all.ps1 -NoBuild
```

### 常用 GPU 的 CUDA 架构号

| GPU | 架构号 |
|-----|--------|
| RTX 4090 | 89 |
| RTX 4080 | 89 |
| RTX 4070 Ti | 89 |
| RTX 3090 | 86 |
| RTX 3080 | 86 |
| RTX 3070 | 86 |
| A100 | 80 |
| V100 | 70 |

修改方式：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86"
```

---

## CLI 使用

### 纯文本合成

**Linux：**

```bash
./build/fish-server \
  --model-dir models/s2-pro-fp16 \
  --text "今天天气真好，我们出去散步吧。" \
  --output output/speech.wav \
  --max-tokens 256 \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 50 \
  --seed 42
```

**Windows（PowerShell）：**

```powershell
.\build_win\Release\fish-server.exe `
  --model-dir models\s2-pro-fp16 `
  --text "今天天气真好，我们出去散步吧。" `
  --output output\speech.wav `
  --max-tokens 256 --seed 42
```

> Windows 下中文直接传参即可，exe 内部使用 `wmain` 从 OS 获取 UTF-16 参数，无需额外设置编码。

### 参考音频（Voice Cloning）

**Linux：**

```bash
./build/fish-server \
  --model-dir models/s2-pro-fp16 \
  --ref-audio example/vo_LLZAQ001_4_nahida_03.wav \
  --ref-text example/vo_LLZAQ001_4_nahida_03.lab \
  --text "Traveler, shall we go on an adventure today?" \
  --output output/cloned.wav \
  --max-tokens 256
```

**Windows：**

```powershell
.\build_win\Release\fish-server.exe `
  --model-dir models\s2-pro-fp16 `
  --ref-audio example\vo_LLZAQ001_4_nahida_03.wav `
  --ref-text example\vo_LLZAQ001_4_nahida_03.lab `
  --text "Traveler, shall we go on an adventure today?" `
  --output output\cloned.wav --max-tokens 256
```

- `--ref-audio`：参考音频 WAV 文件（任意采样率，内部自动重采样到 44.1kHz）
- `--ref-text`：参考音频的转录文本（`.lab` 文件或直接传文本字符串）
- `--ref-codes`：预编码的 VQ codes 文件（可选，跳过实时编码）

### 音频编解码（不经过 DualAR）

```bash
# 编码：WAV → VQ codes
./build/fish-server \
  --model-dir models/s2-pro-fp16 \
  --encode-audio input.wav \
  --output-codes codes.bin

# 解码：VQ codes → WAV
./build/fish-server \
  --model-dir models/s2-pro-fp16 \
  --decode-codes codes.bin \
  --output decoded.wav
```

### 选择精度

```bash
# FP16（默认，自动检测）
./build/fish-server --model-dir models/s2-pro-fp16 --text "你好"

# INT8（节省显存 ~50%）
./build/fish-server --model-dir models/s2-pro-fp16 --dtype int8 --text "你好"

# 直接用 INT8 模型目录
./build/fish-server --model-dir models/s2-pro-int8-w8a8-g256 --text "你好"
```

### 完整参数列表

```
  -m, --model-dir    模型目录（默认: checkpoints/s2-pro）
  -t, --text         要合成的文本
      --max-tokens   最大生成 code frames（默认: 512）
      --temperature  采样温度（默认: 0.7）
      --top-p        Top-p 采样（默认: 0.9）
      --top-k        Top-k 采样（默认: 50）
      --seed         随机种子（默认: 42）
  -o, --output       输出 WAV 路径（默认: output/speech.wav）
      --ref-audio    参考音频 WAV 文件
      --ref-codes    预编码 VQ codes 文件
      --ref-text     参考音频转录文本
      --prompt-file  预构建 prompt 文件（高级用法）
      --tokenize-only 仅打印 token IDs
      --encode-audio 编码音频为 VQ codes
      --output-codes VQ codes 输出路径
      --decode-codes 解码 VQ codes 为 WAV
      --server       启动 HTTP 服务器模式
      --host         服务器绑定地址（默认: 0.0.0.0）
      --port         服务器端口（默认: 8080）
      --dtype        精度：fp16, int8。默认自动检测
  -h, --help         打印帮助
```

---

## HTTP Server

```bash
./build/fish-server \
  --model-dir models/s2-pro-fp16 \
  --server \
  --host 0.0.0.0 \
  --port 8080
```

服务启动后，GPU 模型常驻内存，可处理并发请求（通过 `inference_mutex_` 串行执行）。

---

## API 接口

### 健康检查

```bash
curl http://localhost:8080/health
# → {"status":"ok"}
```

### 纯文本 TTS（非流式）

```bash
curl -X POST http://localhost:8080/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "今天天气真好",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "seed": 42,
    "response_format": "wav"
  }'
```

返回 JSON：

```json
{
  "audio": "<base64 encoded WAV>",
  "audio_format": "wav",
  "sample_rate": 44100,
  "num_samples": 65536,
  "duration_s": 1.49,
  "profiling": { ... }
}
```

`response_format` 可选值：`wav`（默认）、`pcm_f32`。

### 纯文本 TTS（SSE 流式）

```bash
curl -X POST http://localhost:8080/v1/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "今天天气真好", "max_new_tokens": 256}'
```

SSE 事件类型：

| event type | 含义 |
|-----------|------|
| `progress` | AR 解码进度 |
| `audio` | 音频块（base64 pcm_f32） |
| `done` | 推理完成，包含 profiling |
| `error` | 错误信息 |

### 参考音频 TTS（非流式）

```bash
REF_AUDIO=$(base64 -w0 example/vo_LLZAQ001_4_nahida_03.wav)
REF_TEXT=$(cat example/vo_LLZAQ001_4_nahida_03.lab)

curl -X POST http://localhost:8080/v1/tts/with-ref \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"Traveler, shall we go on an adventure today?\",
    \"ref_audio\": \"$REF_AUDIO\",
    \"ref_audio_format\": \"wav\",
    \"ref_text\": \"$REF_TEXT\",
    \"max_new_tokens\": 256
  }"
```

### 参考音频 TTS（SSE 流式，支持句子级 chunking）

```bash
curl -X POST http://localhost:8080/v1/tts/with-ref/stream \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"Traveler, shall we go on an adventure today? ...\",
    \"ref_audio\": \"$REF_AUDIO\",
    \"ref_audio_format\": \"wav\",
    \"ref_text\": \"$REF_TEXT\",
    \"max_new_tokens\": 256,
    \"chunk_length\": 1,
    \"history_frames\": 96
  }"
```

额外参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_length` | `0` | `>0` 时按句子分割文本，逐句推理。推荐 ~120 |
| `history_frames` | `96` | 跨 chunk 携带的历史 VQ frames 数量 |
| `ref_audio_format` | `auto` | `wav`、`pcm_f32`、`auto` |
| `ref_sample_rate` | 模型采样率 | 参考音频的原始采样率 |

### 服务器信息

```bash
curl http://localhost:8080/v1/info
# → {"engine":"fish-audio-cpp","version":"0.1.0","backend":"CUDA"}
```

---

## Python 测试脚本

目录 `scripts/test_api.py` 提供了一个完整的流式 TTS + 实时播放客户端：

```bash
# 先启动服务器
./build/fish-server --model-dir models/s2-pro-fp16 --server --port 8080 &

# 运行测试脚本（需要 ffplay）
pip install requests
python3 scripts/test_api.py \
  --host 127.0.0.1 --port 8080 \
  --text "你好，这是一个流式语音合成测试。" \
  --ref-wav example/vo_LLZAQ001_4_nahida_03.wav \
  --ref-text example/vo_LLZAQ001_4_nahida_03.lab \
  --chunk-length 120 \
  --history-frames 96
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | 服务器地址 |
| `--port` | `8080` | 服务器端口 |
| `--text` | 中英文混合测试文本 | 要合成的文本 |
| `--ref-wav` | nahida 示例 | 参考音频文件 |
| `--ref-text` | nahida 示例 | 参考文本文件 |
| `--chunk-length` | `120` | 句子级 chunk 大小（UTF-8 字节），0=不分割 |
| `--history-frames` | `96` | 跨 chunk 历史帧数 |
| `--max-new-tokens` | `300` | 最大生成帧数 |
| `--no-play` | false | 不使用 ffplay 播放 |
| `--repeat` | `1` | 重复请求次数 |

---

## 显存参考

| 精度 | 模型权重 | 总 GPU 占用 | 最低 GPU 建议 |
|------|---------|------------|--------------|
| FP16 | ~8.7 GB (DualAR) + ~1.5 GB (DAC) | ~12-13 GB | RTX 4080 (16GB) |
| INT8 | ~4.3 GB (DualAR) + ~1.5 GB (DAC) | ~7-8 GB | RTX 3080 (10GB) |

> RTX 4090 (24GB) 上实测 FP16 总占用约 10.5 GB（含 KV cache + workspace）。

### 调整 KV Cache 大小

如果 prompt 很长（参考音频 + 长文本），可能需要更多 KV cache blocks。BlockManager 会自动计算：

```cpp
max_blocks = (max_tokens + prompt_context_len + 15) / 16 + 16;
```

可以通过减小 `--max-tokens` 来降低 KV cache 需求。

---

## 项目结构

```
fish-audio-cpp/
├── src/
│   ├── main.cc                    # CLI + Server 入口
│   ├── engine/
│   │   ├── dual_ar_engine.cc/.h   # Qwen3 DualAR 推理（文本→VQ codes）
│   │   ├── dac_engine.cc/.h       # DAC codec 推理（VQ codes→音频）
│   │   ├── block_manager.cc/.h    # Paged KV Cache 管理
│   │   ├── inference_pipeline.cc/.h # 推理管线编排
│   │   └── scheduler.cc/.h       # 批处理调度器
│   ├── kernels/                   # CUDA kernels
│   ├── model/                     # 模型加载/配置/张量定义
│   ├── server/                    # HTTP 服务器 + 路由
│   ├── tokenizer/                 # BPE Tokenizer（via sentencepiece）
│   └── utils/                     # CUDA 工具宏
├── tools/
│   ├── step1_convert_fp16.py      # PyTorch→FP16 .bin 转换
│   ├── step2_export_w8a8.py       # FP16→INT8（AWQ+GPTQ）导出
│   ├── collect_calibration_w8a8.py # INT8 校准数据采集
│   ├── merge_calibration_dumps.py # 校准数据合并
│   └── quant_utils.py             # 量化工具函数
├── scripts/
│   ├── test_all.sh                # 全模型测试矩阵（Linux）
│   ├── test_all.ps1               # 全模型测试矩阵（Windows PowerShell）
│   ├── build_windows.ps1          # Windows 一键构建脚本
│   ├── build_and_test.sh          # CI 构建+测试
│   └── test_api.py                # Python SSE 流式客户端
├── models/                        # 模型目录（符号链接布局）
├── checkpoints/                   # 原始模型文件（.bin, config, tokenizer）
├── example/                       # 示例参考音频
└── tests/                         # 单元测试
```

---

## License

本项目仅用于学习和研究目的。原模型 [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) 遵循其自身的许可协议。
