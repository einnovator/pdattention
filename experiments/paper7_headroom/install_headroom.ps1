param(
    [string]$Venv = "D:\git\rd\.venv-headroom-037",
    [string]$Version = "0.37.0"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "$Venv\Scripts\python.exe")) {
    py -3.10 -m venv $Venv
}

$python = "$Venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install "headroom-ai[proxy,evals]==$Version"
& $python -m pip check
& $python -c "import importlib.metadata as m; print('headroom-ai', m.version('headroom-ai'))"
