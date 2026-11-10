[CmdletBinding()]
param(
    [switch]$NoBrowser,
    [switch]$NoAdb
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $projectRoot "tmp\runtime"
$pidPath = Join-Path $runtimeDir "kebbi-server.pid"
$stdoutPath = Join-Path $runtimeDir "uvicorn.out.log"
$stderrPath = Join-Path $runtimeDir "uvicorn.err.log"
$healthUrl = "http://127.0.0.1:8000/health"

New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

$pythonCandidates = @(
    (Join-Path $projectRoot ".venv_phase3\Scripts\python.exe"),
    (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1)
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

if (-not $pythonCandidates) {
    throw "找不到 Python；請先執行 setup.bat 建立 .venv_phase3。"
}
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot "models\metadata.json"))) {
    throw "找不到正式模型；請先執行 setup.bat 或 python -m src.train_final。"
}
$pythonExe = @($pythonCandidates)[0]

function Test-LocalServer {
    try {
        $response = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        return $response.status -eq "ok"
    } catch {
        return $false
    }
}

if (-not (Test-LocalServer)) {
    Write-Host "正在背景啟動肌少症檢測伺服器…"
    $server = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList @("-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000") `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -WindowStyle Hidden `
        -PassThru
    Set-Content -LiteralPath $pidPath -Value $server.Id -Encoding ascii

    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 1
        if (Test-LocalServer) {
            $ready = $true
            break
        }
        if ($server.HasExited) { break }
    }
    if (-not $ready) {
        if (-not $server.HasExited) {
            Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
        }
        throw "伺服器未能在 30 秒內啟動，請查看 $stderrPath"
    }
} else {
    Write-Host "伺服器已在執行，直接沿用。"
}

Write-Host "伺服器正常：http://127.0.0.1:8000"

if (-not $NoAdb) {
    $adbCandidates = @(
        (Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"),
        (Get-Command adb -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1)
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

    if (-not $adbCandidates) {
        Write-Warning "找不到 ADB。伺服器已啟動，但無法自動設定凱比 USB 轉接。"
    } else {
        $adbExe = @($adbCandidates)[0]
        & $adbExe start-server | Out-Null
        $deviceLines = & $adbExe devices | Select-String -Pattern "^\S+\s+device$"
        if ($deviceLines.Count -eq 1) {
            & $adbExe reverse tcp:8000 tcp:8000 | Out-Null
            & $adbExe shell am start -a android.intent.action.VIEW -d "http://127.0.0.1:8000/" | Out-Null
            Write-Host "凱比已連線，已設定 ADB reverse 並開啟正式檢測頁。"
        } elseif ($deviceLines.Count -eq 0) {
            Write-Warning "目前沒有偵測到凱比。請開機並接上 USB 後，再執行 start_kebbi.bat。"
        } else {
            Write-Warning "偵測到多台 Android 裝置，為避免開錯裝置，未自動設定 ADB reverse。"
        }
    }
}

if (-not $NoBrowser) {
    Start-Process "http://localhost:8000/control"
    Write-Host "已開啟電腦控制台。"
}
