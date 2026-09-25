@echo off
chcp 65001 >nul
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [错误] 未找到 Python 虚拟环境: %VENV_PY%
    pause
    exit /b 1
)
echo ========================================
echo   今日头条自动发文系统 v2.0
echo ========================================
echo.
"%VENV_PY%" toutiao.py
pause
