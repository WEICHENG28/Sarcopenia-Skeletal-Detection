@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist "models\metadata.json" (
    echo 找不到模型檔，請先執行 setup.bat
    pause
    exit /b 1
)

echo 啟動伺服器… 瀏覽器開啟 http://localhost:8000
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
pause
