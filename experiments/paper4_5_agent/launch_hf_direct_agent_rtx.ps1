param(
    [string]$RepoRoot = "C:\Users\killu\git\rd\pdattention-paper4",
    [string]$Python = "C:\Users\killu\.venvs\ccpu-cuda\Scripts\python.exe",
    [string]$Model = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    [string]$Revision = "2e1fd397ee46e1388853d2af2c993145b0f1098a",
    [int]$Port = 18123,
    [int]$MaximumUsedGpuMiB = 2500,
    [switch]$InstallDependencies
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $RepoRoot)) {
    throw "Paper 4 repository is absent: $RepoRoot"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Prepared CUDA Python is absent: $Python"
}

$usedText = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi failed; the HF CUDA lane is not ready."
}
$usedMiB = [int](($usedText | Select-Object -First 1).Trim())
if ($usedMiB -gt $MaximumUsedGpuMiB) {
    throw "GPU is still occupied ($usedMiB MiB used; limit $MaximumUsedGpuMiB MiB)."
}

if ($InstallDependencies) {
    & $Python -m pip install --disable-pip-version-check -e "$RepoRoot[hf-runtime]"
    if ($LASTEXITCODE -ne 0) {
        throw "HF runtime dependency installation failed."
    }
}

& $Python -c "import accelerate, torch, transformers; assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA/Transformers preflight failed. Re-run with -InstallDependencies."
}

$env:HF_HUB_OFFLINE = "1"
$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
Set-Location -LiteralPath $RepoRoot
& $Python -m experiments.paper4_5_agent.serve_hf_agent_pra `
    --model $Model `
    --revision $Revision `
    --host 127.0.0.1 `
    --port $Port `
    --wire-tail-tokens 32 `
    --device-map cuda:0 `
    --chat-template-profile native
