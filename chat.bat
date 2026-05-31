@echo off
chcp 65001 >nul

REM ── 语言检测 ──
set T_ZH=0
reg query "HKCU\Control Panel\International" /v sLanguage 2>nul | find "zh" >nul && set T_ZH=1

set DIR=%~dp0
set VENV=%DIR%.venv
if not exist "%VENV%" (
    if "%T_ZH%"=="1" (
        echo 虚拟环境不存在，请先运行: setup.bat
    ) else (
        echo Virtual environment not found. Run first: setup.bat
    )
    pause
    exit /b 1
)
call "%VENV%\Scripts\activate.bat"
python "%DIR%chat.py" %*
pause
