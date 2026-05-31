# Hy-MT2 OpenVINO 转换与部署

腾讯混元 Hy-MT2 多语言翻译模型的 OpenVINO GPU 推理方案。

> **免责声明**：本项目所有代码由 AI 生成，在作者设备上成功运行。**在您的设备上运行之前，请务必进行审查。** 作者不对因使用本项目造成的任何直接或间接损失承担责任。

## 支持模型

| 模型 | 架构 | 参数量 | OpenVINO 转换 | 说明 |
|------|------|--------|:-------------:|------|
| Hy-MT2-1.8B | HunYuanDenseV1ForCausalLM | 1.79B | ✅ 已验证 | FP32: 3.4GB / INT8: 1.9GB / INT4: 1.0GB |
| Hy-MT2-7B | HunYuanDenseV1ForCausalLM | 6.9B | ✅ 改路径即可 | 同架构，需约 14GB (FP32) |
下载地址：[HuggingFace](https://huggingface.co/collections/tencent/hy-mt2) / [ModelScope](https://modelscope.cn/collections/Tencent-Hunyuan/Hy-MT2)

## 目录结构

```
Hy-MT2/
├── README.md              ← 本文件
├── setup.sh               ← 一键初始化环境 (Linux/macOS)
├── setup.bat              ← 一键初始化环境 (Windows)
├── convert_to_ov.py       ← 模型转换脚本
├── chat.py                ← 终端翻译聊天
├── chat.sh                ← 一键启动脚本 (Linux/macOS)
├── chat.bat               ← 一键启动脚本 (Windows)
├── api_server.py          ← OpenAI 兼容 API 服务
├── .venv/                 ← Python 虚拟环境（setup.sh 自动创建）
├── 1.8B/                  ← 原始模型（需自行下载）
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
└── 1.8B-ov/               ← 转换后的 OpenVINO 模型
    ├── openvino_model.xml
    ├── openvino_model.bin
    ├── config.json
    └── tokenizer.json
```

## 快速开始

### 0. 环境初始化（首次运行）

**Linux / macOS：**
```bash
cd Hy-MT2
bash setup.sh
```

**Windows：**
```cmd
cd Hy-MT2
setup.bat
```

脚本会：① 检测 Python 版本（Windows 下自动下载安装 Python 3.14） → ② **询问选择 PyTorch 后端**（CPU 通用 / XPU Intel GPU 加速） → ③ 创建 `.venv` 虚拟环境 → ④ 安装全部依赖。

> 没有 Intel Arc 显卡选 **CPU** 即可，OpenVINO 会自动优化 CPU 推理。

### 1. 下载原始模型

```bash
# 从 HuggingFace 下载
pip install huggingface-hub
huggingface-cli download tencent/Hy-MT2-1.8B --local-dir 1.8B

# 或从 ModelScope
pip install modelscope
modelscope download --model Tencent-Hunyuan/Hy-MT2-1.8B --local_dir 1.8B
```

### 2. 转换为 OpenVINO

```bash
source .venv/bin/activate

# FP32（默认，约 3.4GB）
python convert_to_ov.py

# INT8 量化（约 1.9GB，质量几乎无损）
python convert_to_ov.py --weight-format int8

# INT4 量化（约 1.0GB，速度最快）
python convert_to_ov.py --weight-format int4

# 指定路径
python convert_to_ov.py --model 7B --output 7B-ov --weight-format int8
```

转换约需 **3-5 分钟**，输出文件在 `1.8B-ov/`。

> **转换原理**：模型使用了 `torch.vmap` 构建因果掩码，该操作无法被 `torch.jit.trace` 追踪。脚本通过自定义包装器手动构造 4D causal mask 绕过此限制。
>
> **KV Cache**：转换时设置了 `use_cache=False`，因为 `DynamicCache/past_key_values` 等运行时对象无法被 `jit.trace` 追踪。这会导致生成长文本时每步重新计算全部注意力（约 2-3 倍开销）。短文本翻译（< 50 tokens）几乎不受影响。如需支持长文本生成，可自行在 wrapper 中手动传递 KV 张量。

### 3. 运行

#### 终端翻译聊天（推荐）

```bash
# 默认（1.8B FP32）
bash chat.sh

# 指定 INT4 量化模型
bash chat.sh --ov-path 1.8B-ov-int4

# 或用环境变量
export HY_MT2_OV_PATH=1.8B-ov-int4
bash chat.sh
```

> 切换模型只需改变 `--ov-path` 指向不同 OpenVINO 目录，无需修改代码。

#### API 服务

```bash
cd Hy-MT2
source .venv/bin/activate
python api_server.py --port 8000
```

启动后可用 OpenAI 客户端调用：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="Hy-MT2-1.8B",
    messages=[{"role": "user", "content": "将以下文本翻译为英语：\n\n今天天气真好。"}],
)
print(resp.choices[0].message.content)
```

## 功能特性

### 终端翻译聊天 (`chat.py`)

```
==================================================
  Hy-MT2 翻译终端
  GPU: Intel Arc iGraphics | OpenVINO
==================================================
  直接输入文本 → 自动检测翻译方向
  //en 文本    → 强制译英
  //zh 文本    → 强制译中
  /help        → 帮助
  /exit        → 退出
==================================================
```

- **自动语言检测**：含中文→译英，不含中文→译中
- **强制方向**：`//en` / `//zh` 前缀指定目标语言
- **支持 33 种语言**：中、英、日、韩、法、德、西、俄、阿 等
- **翻译速度**：~1-2 秒/次（GPU Intel Arc iGraphics）

### API 服务 (`api_server.py`)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | OpenAI 兼容聊天补全 |
| `/v1/translate` | POST | 原生翻译接口 |
| `/v1/models` | GET | 模型列表 |
| `/` | GET | 服务信息 |

- **流式输出**：SSE 逐 token 推送
- **采样控制**：temperature、top_p、top_k、repetition_penalty
- **详细日志**：每个请求记录 prompt、参数、耗时、结果

## 转换 7B 模型

Hy-MT2-7B 与 1.8B 架构完全相同，直接用 `--model` 指定路径：

```bash
cd Hy-MT2
source .venv/bin/activate
python convert_to_ov.py --model 7B --output 7B-ov

# 或 INT8 量化
python convert_to_ov.py --model 7B --output 7B-ov --weight-format int8
```

同时启动时用 `--ov-path` 指向对应目录即可：

```bash
bash chat.sh --ov-path 7B-ov
```

## 推荐参数

翻译是高精度任务，推荐使用**贪心解码**以获得最稳定的结果：

| 参数 | 推荐值 | 说明 |
|------|:------:|------|
| `temperature` | **0** | 贪心解码，翻译最准确 |
| `top_p` | 1.0 | temperature=0 时不生效 |
| `top_k` | -1（关闭） | temperature=0 时不生效 |
| `repetition_penalty` | **1.0**（关闭） | Hy-MT2 本身几乎不重复 |
| `max_tokens` | **512** | 一般翻译 50 token 内完成 |

> 如果想稍微增加多样性，可设 `temperature=0.1`，不建议超过 0.2。

## 环境要求

- **Python**: 3.10+
- **PyTorch**: 2.x
- **OpenVINO**: 2026.2+
- **transformers**: 4.57+
- **GPU (推荐)**: Intel Arc 系列 (iGPU/dGPU)，需 OpenVINO GPU 插件 → ~1-2 秒/次
- **CPU (可用)**: 任何 x86_64 处理器，OpenVINO 自动优化 → ~3-5 秒/次
- **内存**: 1.8B 需 ~5GB，7B 需 ~14GB

## 文件说明

| 文件 | 用途 |
|------|------|
| `setup.sh` | 一键初始化环境（Linux/macOS，创建 venv + 安装依赖） |
| `setup.bat` | 一键初始化环境（Windows，自动装 Python + 创建 venv） |
| `convert_to_ov.py` | PyTorch → OpenVINO IR 转换 |
| `chat.py` | 终端交互式翻译聊天 |
| `chat.sh` | 一键启动脚本（Linux/macOS，自动找 venv） |
| `chat.bat` | 一键启动脚本（Windows，自动找 venv） |
| `api_server.py` | FastAPI OpenAI 兼容翻译服务 |
| `1.8B/` | 原始 PyTorch 模型（需下载） |
| `1.8B-ov/` | 转换后的 OpenVINO 模型 |

## 常见问题

**Q: 转换失败，报 torch.vmap 相关错误？**
A: 这是已知问题，`convert_to_ov.py` 已通过包装器绕过。如果还有问题，检查 transformers 版本 ≥ 4.57.6。

**Q: 翻译结果乱码？**
A: 确保 temperature=0（贪心解码），翻译任务推荐确定性输出。

**Q: 支持 NPU 吗？**
A: 不支持。1.8B FP32 模型约 3.4GB，超过 NPU 内存限制。

**Q: 支持流式翻译吗？**
A: 终端和 API 均支持流式逐 token 输出。

**Q: 转换后的模型为什么没有 KV Cache？影响大吗？**
A: 因为 `DynamicCache` 是运行时对象，`torch.jit.trace` 无法追踪。转换时设置了 `use_cache=False`，每步重新计算全部注意力。对于翻译场景（短文本 < 50 tokens），影响很小。如需支持长文本生成，可在 wrapper 中手动传递 KV 张量。

---

[English](#english) | [中文](#hy-mt2-openvino-转换与部署)

---

# <a id="english"></a>Hy-MT2 OpenVINO Inference

Tencent Hy-MT2 multilingual translation model with OpenVINO GPU acceleration.

> **Disclaimer**: All code in this project was AI-generated and tested on the author's device. **Please review all code before running it on your system.** The author assumes no responsibility for any direct or indirect damages resulting from the use of this project.

## Supported Models

| Model | Architecture | Params | OpenVINO | Details |
|-------|-------------|--------|:--------:|---------|
| Hy-MT2-1.8B | HunYuanDenseV1ForCausalLM | 1.79B | ✅ Verified | FP32: 3.4GB / INT8: 1.9GB / INT4: 1.0GB |
| Hy-MT2-7B | HunYuanDenseV1ForCausalLM | 6.9B | ✅ Change path | Same architecture, ~14GB (FP32) |

Download: [HuggingFace](https://huggingface.co/collections/tencent/hy-mt2) / [ModelScope](https://modelscope.cn/collections/Tencent-Hunyuan/Hy-MT2)

## Quick Start

### 0. Setup (first time)

**Linux / macOS:**
```bash
cd Hy-MT2
bash setup.sh
```

**Windows:**
```cmd
cd Hy-MT2
setup.bat
```

The script will: ① Check Python version → ② **Ask for PyTorch backend** (CPU / Intel XPU) → ③ Create `.venv` → ④ Install all dependencies.

### 1. Download Model

```bash
# From HuggingFace
huggingface-cli download tencent/Hy-MT2-1.8B --local-dir 1.8B

# From ModelScope
modelscope download --model Tencent-Hunyuan/Hy-MT2-1.8B --local_dir 1.8B
```

### 2. Convert to OpenVINO

```bash
source .venv/bin/activate

# FP32 (default, ~3.4GB)
python convert_to_ov.py

# INT8 quantization (~1.9GB)
python convert_to_ov.py --weight-format int8

# INT4 quantization (~1.0GB, fastest)
python convert_to_ov.py --weight-format int4
```

Conversion takes **3-5 minutes**. Output goes to `1.8B-ov/`.

### 3. Run

**Terminal chat（recommended）：**
```bash
bash chat.sh                    # default (1.8B FP32)
bash chat.sh --ov-path 1.8B-ov-int4  # use INT4 model
export HY_MT2_OV_PATH=1.8B-ov-int4
bash chat.sh
```

**API server：**
```bash
source .venv/bin/activate
python api_server.py --port 8000
```

Then use any OpenAI client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="Hy-MT2-1.8B",
    messages=[{"role": "user", "content": "Translate into English:\n\n今天天气真好。"}],
)
print(resp.choices[0].message.content)
```

## Features

- **Terminal chat** with auto language detection (Chinese→English, English→Chinese)
- **OpenAI-compatible API** with SSE streaming
- **33 languages** supported
- **Quantization**: FP32 / INT8 / INT4
- **Device auto-detection**: GPU preferred, CPU fallback

## Recommended Parameters

| Parameter | Value | Note |
|-----------|:-----:|------|
| `temperature` | **0** | Greedy decoding, most accurate for translation |
| `top_p` | 1.0 | Not used when temperature=0 |
| `repetition_penalty` | **1.0** (off) | Hy-MT2 rarely repeats |
| `max_tokens` | **512** | Most translations < 50 tokens |

## Requirements

- **Python**: 3.10+
- **OpenVINO**: 2026.2+
- **transformers**: 4.57+
- **GPU (recommended)**: Intel Arc iGPU/dGPU → ~1-2s per translation
- **CPU (fallback)**: Any x86_64 → ~3-5s per translation
