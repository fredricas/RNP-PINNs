param(
    [string]$RunDir = '',
    [string]$PythonExe = ''
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $RunDir = Join-Path $PSScriptRoot "outputs\standard_formal_mean3_$stamp"
}
$RunDir = [System.IO.Path]::GetFullPath($RunDir)
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}
$PythonExe = [System.IO.Path]::GetFullPath($PythonExe)
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$Supervisor = Join-Path $PSScriptRoot 'run_formal_supervisor.ps1'
$arguments = (
    "-NoProfile -ExecutionPolicy Bypass -File `"$Supervisor`" " +
    "-RunDir `"$RunDir`" -PythonExe `"$PythonExe`""
)
$info = New-Object System.Diagnostics.ProcessStartInfo
$info.FileName = 'powershell.exe'
$info.Arguments = $arguments
$info.UseShellExecute = $true
$info.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
$process = [System.Diagnostics.Process]::Start($info)

$record = [ordered]@{
    run_dir = $RunDir
    supervisor_pid = $process.Id
    started_at = (Get-Date).ToString('o')
    mode = 'formal'
    workers = 2
    bootstrap_replicates = 50
    data_representation = 'mean and sample SD of three independent experimental replicates'
    multistart_fits = 10
    python_executable = $PythonExe
}
$record | ConvertTo-Json | Set-Content -LiteralPath (
    Join-Path $PSScriptRoot 'background_task.json'
) -Encoding UTF8

Write-Output ("RUN_DIR={0}" -f $RunDir)
Write-Output ("SUPERVISOR_PID={0}" -f $process.Id)
