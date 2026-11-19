@echo off
chcp 65001 >nul
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_kebbi_session.ps1"
if errorlevel 1 (
    echo.
    echo 啟動失敗，請查看上方訊息或 tmp\runtime\uvicorn.err.log。
)
pause
