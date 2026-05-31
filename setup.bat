@echo off
chcp 65001 >nul

REM ── 语言选择 ──
echo ========================================
echo   Hy-MT2 Setup
echo ========================================
echo.
echo Select language / 选择语言:
echo   1) English
echo   2) 中文
echo.
set /p lang_choice="Enter [1/2] (default 1): "
if "%lang_choice%"=="" set lang_choice=1

if "%lang_choice%"=="2" (
    set T_ZH=1
    title Hy-MT2 环境初始化
    echo ⚠  所有代码由 AI 生成，在您的设备上运行之前，请务必进行审查。
) else (
    set T_ZH=0
    title Hy-MT2 Setup
    echo ⚠  AI-generated code. Please review before running on your system.
)

REM ── 工具函数 ──
goto :main

:echo_zh
if "%T_ZH%"=="1" echo %~1
goto :eof

:echo_en
if "%T_ZH%"=="0" echo %~1
goto :eof

:echo_both
if "%T_ZH%"=="1" (echo %~1) else (echo %~2)
goto :eof

:main
echo.

REM ── 检测 Python ──
set PYTHON=python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    call :echo_both "✗ 未检测到 Python，正在下载 Python 3.14 ..." "✗ Python not found, downloading Python 3.14 ..."
    echo.
    curl -L -o "%TEMP%\python-3.14.0-amd64.exe" https://www.python.org/ftp/python/3.14.0/python-3.14.0-amd64.exe
    if %ERRORLEVEL% NEQ 0 (
        call :echo_both "下载失败，请手动安装 Python 3.10+ 后重试" "Download failed, please install Python 3.10+ manually"
        call :echo_both "下载地址: https://www.python.org/downloads/" "Download: https://www.python.org/downloads/"
        pause
        exit /b 1
    )
    call :echo_both "安装 Python（请勾选 Add Python to PATH）..." "Installing Python (check Add Python to PATH) ..."
    start /wait "%TEMP%\python-3.14.0-amd64.exe" /quiet InstallAllUsers=1 PrependPath=1
    if %ERRORLEVEL% NEQ 0 (
        call :echo_both "静默安装失败，尝试手动安装..." "Silent install failed, trying manual install..."
        start "" "%TEMP%\python-3.14.0-amd64.exe"
        call :echo_both "请手动完成安装后，按任意键继续..." "Please complete installation manually, then press any key..."
        pause
    )
    for /f "tokens=2*" %%i in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "PATH=%%j;%PATH%"
)

python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    call :echo_both "Python 安装失败，请手动安装 Python 3.10+" "Python installation failed, please install Python 3.10+ manually"
    pause
    exit /b 1
)

REM ── 选择 PyTorch 后端 ──
echo.
call :echo_both "选择 PyTorch 安装方式：" "Select PyTorch backend:"
call :echo_both "  1) CPU  — 通用，任何电脑都能用" "  1) CPU  — Universal, works on any machine"
call :echo_both "  2) XPU  — Intel GPU 加速（需 Intel Arc 显卡）" "  2) XPU  — Intel GPU acceleration (requires Intel Arc GPU)"
echo.
set /p choice="> "
if "%choice%"=="" set choice=1

if "%choice%"=="2" (
    set PYTORCH_INDEX=https://download.pytorch.org/whl/xpu
    set BACKEND_LABEL=XPU
) else (
    set PYTORCH_INDEX=https://download.pytorch.org/whl/cpu
    set BACKEND_LABEL=CPU
)

REM ── 创建虚拟环境 ──
set VENV=%~dp0.venv
if exist "%VENV%" (
    echo.
    call :echo_both "虚拟环境已存在，是否重建？" "Virtual environment exists. Rebuild?"
    set /p rebuild="[y/N]: "
    if /i "!rebuild!"=="y" rmdir /s /q "%VENV%"
)

if not exist "%VENV%" (
    echo.
    call :echo_both "创建虚拟环境 ..." "Creating virtual environment ..."
    python -m venv "%VENV%"
    if %ERRORLEVEL% NEQ 0 (
        call :echo_both "创建虚拟环境失败" "Failed to create virtual environment"
        pause
        exit /b 1
    )
    call :echo_both "✓ 虚拟环境创建完成" "✓ Virtual environment created"
)

call "%VENV%\Scripts\activate.bat"

REM ── 升级 pip ──
echo.
call :echo_both "升级 pip ..." "Upgrading pip ..."
python -m pip install --upgrade pip wheel setuptools

REM ── 安装 PyTorch ──
echo.
call :echo_both "安装 PyTorch (%BACKEND_LABEL%) ..." "Installing PyTorch (%BACKEND_LABEL%) ..."
pip install torch torchvision torchaudio --index-url %PYTORCH_INDEX%

REM ── 安装其他依赖 ──
echo.
call :echo_both "安装 OpenVINO + Transformers + 其他依赖 ..." "Installing OpenVINO + Transformers + other dependencies ..."
pip install openvino openvino-tokenizers optimum-intel nncf
pip install transformers tokenizers sentencepiece safetensors
pip install numpy tqdm
pip install fastapi uvicorn pydantic

REM ── 验证 ──
echo.
echo ========================================
if "%T_ZH%"=="1" (echo   环境验证) else (echo   Verification)
echo ========================================
echo.
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import openvino; print(f'OpenVINO {openvino.__version__}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
python -c "import nncf; print(f'NNCF {nncf.__version__}')"

echo.
echo ========================================
if "%T_ZH%"=="1" (echo   ✅ 环境初始化完成!) else (echo   ✅ Setup complete!)
echo ========================================
echo.
if "%T_ZH%"=="1" (
    echo 激活环境: .venv\Scripts\activate
    echo 转换模型: python convert_to_ov.py
    echo 启动聊天: python chat.py
    echo 启动 API: python api_server.py
) else (
    echo Activate: .venv\Scripts\activate
    echo Convert:  python convert_to_ov.py
    echo Chat:     python chat.py
    echo API:      python api_server.py
)
echo.
pause
