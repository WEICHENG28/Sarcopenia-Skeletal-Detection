[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pidPath = Join-Path $projectRoot "tmp\runtime\kebbi-server.pid"

if (-not (Test-Path -LiteralPath $pidPath)) {
    Write-Host "找不到由 start_kebbi.bat 啟動的伺服器紀錄；不會停止其他 Python 程序。"
    exit 0
}

$serverPid = [int](Get-Content -LiteralPath $pidPath -Raw).Trim()
$process = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
if ($process) {
    Stop-Process -Id $serverPid
    [void]$process.WaitForExit(5000)
    Write-Host "已停止肌少症檢測伺服器。"
} else {
    Write-Host "伺服器程序已經結束。"
}
Remove-Item -LiteralPath $pidPath -Force
