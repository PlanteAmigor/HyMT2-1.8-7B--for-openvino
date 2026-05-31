#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# ── 语言选择 ──
echo "========================================"
echo "  Hy-MT2 Setup"
echo "========================================"
echo ""
echo "Select language / 选择语言:"
echo "  1) English"
echo "  2) 中文"
echo ""
read -p "Enter [1/2] (default 1): " lang_choice
lang_choice=${lang_choice:-1}

if [ "$lang_choice" = "2" ]; then
    T_PY_FOUND="✓ 检测到 Python"
    T_PY_MISS="✗ 需要 Python 3.10+，请先安装"
    T_SEL_BACKEND="选择 PyTorch 安装方式："
    T_OPT_CPU=" 1) CPU  — 通用，任何电脑都能用"
    T_OPT_XPU=" 2) XPU  — Intel GPU 加速（需 Intel Arc 显卡 + 驱动）"
    T_CHOICE="请输入 [1/2] (默认 1)"
    T_VENV_EXISTS="虚拟环境已存在，是否重建？(重新安装所有依赖)"
    T_REBUILD="重建? [y/N]"
    T_VENV_CREATE="创建虚拟环境 ..."
    T_VENV_DONE="✓ 虚拟环境创建完成"
    T_PIP_UP="升级 pip ..."
    T_INSTALL_TORCH="安装 PyTorch"
    T_INSTALL_OV="安装 OpenVINO + Transformers + 其他依赖 ..."
    T_VERIFY="环境验证"
    T_DONE="✅ 环境初始化完成!"
    T_ACTIVATE="激活环境"
    T_CONVERT="转换模型"
    T_CHAT="启动聊天"
    T_API="启动 API"
else
    T_PY_FOUND="✓ Python detected"
    T_PY_MISS="✗ Python 3.10+ required, please install it first"
    T_SEL_BACKEND="Select PyTorch backend:"
    T_OPT_CPU=" 1) CPU  — Universal, works on any machine"
    T_OPT_XPU=" 2) XPU  — Intel GPU acceleration (requires Intel Arc GPU)"
    T_CHOICE="Enter [1/2] (default 1)"
    T_VENV_EXISTS="Virtual environment exists. Rebuild? (reinstall all dependencies)"
    T_REBUILD="Rebuild? [y/N]"
    T_VENV_CREATE="Creating virtual environment ..."
    T_VENV_DONE="✓ Virtual environment created"
    T_PIP_UP="Upgrading pip ..."
    T_INSTALL_TORCH="Installing PyTorch"
    T_INSTALL_OV="Installing OpenVINO + Transformers + other dependencies ..."
    T_VERIFY="Verification"
    T_DONE="✅ Setup complete!"
    T_ACTIVATE="Activate environment"
    T_CONVERT="Convert model"
    T_CHAT="Start chat"
    T_API="Start API server"
fi

echo ""
echo "========================================"
echo "  Hy-MT2 Setup"
echo "========================================"
echo ""

# ── 免责声明 ──
if [ "$lang_choice" = "2" ]; then
    echo "⚠  所有代码由 AI 生成，在您的设备上运行之前，请务必进行审查。"
else
    echo "⚠  AI-generated code. Please review before running on your system."
fi
echo ""

# ── 检测 Python ──
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if [[ "$ver" =~ ^3\.(1[0-9]) ]]; then
            PYTHON="$cmd"
            echo "$T_PY_FOUND $ver: $(which $cmd)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "$T_PY_MISS"
    exit 1
fi

# ── 选择 PyTorch 后端 ──
echo ""
echo "$T_SEL_BACKEND"
echo "$T_OPT_CPU"
echo "$T_OPT_XPU"
echo ""
read -p "$T_CHOICE: " choice
choice=${choice:-1}

case "$choice" in
    2|xpu*|XPU*) PYTORCH_INDEX="https://download.pytorch.org/whl/xpu"; BACKEND_LABEL="XPU" ;;
    *)           PYTORCH_INDEX="https://download.pytorch.org/whl/cpu";  BACKEND_LABEL="CPU" ;;
esac

# ── 创建虚拟环境 ──
VENV="$DIR/.venv"
if [ -d "$VENV" ]; then
    echo ""
    echo "$T_VENV_EXISTS"
    read -p "$T_REBUILD: " rebuild
    if [[ "$rebuild" == "y" || "$rebuild" == "Y" ]]; then
        rm -rf "$VENV"
    fi
fi

if [ ! -d "$VENV" ]; then
    echo ""
    echo "$T_VENV_CREATE"
    $PYTHON -m venv "$VENV"
    echo "$T_VENV_DONE"
fi

source "$VENV/bin/activate"

# ── 升级 pip ──
echo ""
echo "$T_PIP_UP"
pip install --upgrade pip wheel setuptools

# ── 安装 PyTorch ──
echo ""
echo "$T_INSTALL_TORCH ($BACKEND_LABEL) ..."
pip install torch torchvision torchaudio --index-url "$PYTORCH_INDEX"

# ── 安装其他依赖 ──
echo ""
echo "$T_INSTALL_OV"
pip install openvino openvino-tokenizers optimum-intel nncf
pip install transformers tokenizers sentencepiece safetensors
pip install numpy tqdm
pip install fastapi uvicorn pydantic

# ── 验证 ──
echo ""
echo "========================================"
echo "  $T_VERIFY"
echo "========================================"
echo ""

python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import openvino; print(f'OpenVINO {openvino.__version__}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
python -c "import nncf; print(f'NNCF {nncf.__version__}')"
python -c "
import openvino as ov
core = ov.Core()
print(f'OpenVINO devices: {core.available_devices}')
"

echo ""
echo "========================================"
echo "  $T_DONE"
echo "========================================"
echo ""
echo "$T_ACTIVATE: source .venv/bin/activate"
echo "$T_CONVERT: python convert_to_ov.py"
echo "$T_CHAT: python chat.py"
echo "$T_API: python api_server.py"
echo ""
