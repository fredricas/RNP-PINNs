$ErrorActionPreference = 'Continue'
$TaskFile = Join-Path $PSScriptRoot 'background_task.json'
if (-not (Test-Path -LiteralPath $TaskFile)) {
    Write-Output 'No background task record exists.'
    exit 1
}

$task = Get-Content -LiteralPath $TaskFile -Raw | ConvertFrom-Json
$process = Get-Process -Id $task.supervisor_pid -ErrorAction SilentlyContinue
Write-Output ("Run directory: {0}" -f $task.run_dir)
Write-Output ("Supervisor PID: {0}" -f $task.supervisor_pid)
Write-Output ("Supervisor running: {0}" -f [bool]$process)

$summaryCount = @(
    Get-ChildItem -LiteralPath (Join-Path $task.run_dir 'fits') `
        -Filter summary.json -File -Recurse -ErrorAction SilentlyContinue
).Count
Write-Output ("Completed fit summaries: {0}" -f $summaryCount)

$log = Join-Path $task.run_dir 'supervisor.log'
if (Test-Path -LiteralPath $log) {
    Write-Output 'Recent supervisor log:'
    Get-Content -LiteralPath $log -Tail 15
}
