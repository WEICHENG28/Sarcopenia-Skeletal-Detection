@echo off
chcp 65001 >nul
echo ========================================
echo  肌少症篩檢系統 - 環境安裝
echo ========================================
echo.

echo [1/2] 安裝套件...
pip install -r requirements.txt
if errorlevel 1 goto error

echo.
echo [2/2] 檢查模型檔...
if not exist "models\metadata.json" (
    echo.
    echo   找不到模型檔，正在訓練並封存...
    echo   （首次執行約需 20-40 分鐘）
    echo.
    python -m src.train_final
    if errorlevel 1 goto error
) else (
    echo   模型檔已存在，略過訓練。
)

echo.
echo ========================================
echo  安裝完成，請執行 start_server.bat 啟動
echo ========================================
echo.
pause
goto end

:error
echo.
echo 安裝失敗。請確認：
echo   1. Python 已安裝（執行 python --version 確認）
echo   2. 已在專案根目錄執行本腳本
pause

:end
