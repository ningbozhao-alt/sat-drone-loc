$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
  Write-Host "还没有安装环境。请先在此目录执行："
  Write-Host "  python -m venv .venv"
  Write-Host "  .venv\Scripts\python.exe -m pip install -r requirements.txt"
  exit 1
}

function Test-ServerUp {
  try {
    $response = Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health -TimeoutSec 2
    return $response.StatusCode -eq 200
  } catch {
    return $false
  }
}

if (Test-ServerUp) {
  Write-Host "服务已在运行，正在打开浏览器。"
  Start-Process "http://127.0.0.1:8000"
  exit 0
}

Write-Host "正在启动 http://127.0.0.1:8000"
Start-Process -FilePath $python -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") -WorkingDirectory $PSScriptRoot

$ready = $false
for ($i = 0; $i -lt 40; $i++) {
  if (Test-ServerUp) {
    $ready = $true
    break
  }
  Start-Sleep -Milliseconds 400
}

if (-not $ready) {
  Write-Host "启动超时，请查看新打开的窗口。"
  exit 1
}

Start-Process "http://127.0.0.1:8000"
