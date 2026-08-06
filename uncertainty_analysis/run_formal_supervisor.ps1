param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,
    [string]$PythonExe = 'python'
)

$ErrorActionPreference = 'Continue'
$Python = $PythonExe
$Driver = Join-Path $PSScriptRoot 'run_v5_uq.py'
$RunDir = [System.IO.Path]::GetFullPath($RunDir)
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$SupervisorLog = Join-Path $RunDir 'supervisor.log'

for ($attempt = 1; $attempt -le 20; $attempt++) {
    Add-Content -LiteralPath $SupervisorLog -Encoding UTF8 -Value (
        "{0:o} attempt {1} starting" -f (Get-Date), $attempt
    )
    & $Python -u $Driver --mode formal --sections stability,bootstrap,profile,morris,surface `
        --workers 2 --resume $RunDir 2>&1 | Tee-Object -FilePath $SupervisorLog -Append
    $exitCode = $LASTEXITCODE
    Add-Content -LiteralPath $SupervisorLog -Encoding UTF8 -Value (
        "{0:o} attempt {1} exit code {2}" -f (Get-Date), $attempt, $exitCode
    )
    if ($exitCode -eq 0) {
        exit 0
    }
    Start-Sleep -Seconds 30
}

exit 1
