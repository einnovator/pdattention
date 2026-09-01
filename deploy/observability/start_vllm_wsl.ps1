$ErrorActionPreference = "Stop"

$stdout = Join-Path $env:USERPROFILE "vllm-observed.stdout.log"
$stderr = Join-Path $env:USERPROFILE "vllm-observed.stderr.log"
$arguments = @(
    "-d", "Ubuntu-24.04", "--", "/usr/bin/env", "FOREGROUND=1",
    "/bin/bash", "/mnt/c/Users/killu/start_vllm_observed.sh"
)
$start = @{
    FilePath = "wsl.exe"
    ArgumentList = $arguments
    WindowStyle = "Hidden"
    RedirectStandardOutput = $stdout
    RedirectStandardError = $stderr
    PassThru = $true
}
$process = Start-Process @start
Write-Output "Started hidden WSL vLLM host process $($process.Id)."
