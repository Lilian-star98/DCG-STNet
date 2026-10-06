$ErrorActionPreference = 'Stop'
$formalRoot = 'F:\动态因果'
$formalOutput = Join-Path $formalRoot 'results\formal_v3\study'
$formalPython = Join-Path $formalRoot '.venv-formal\Scripts\python.exe'
$formalPidFile = Join-Path $formalOutput 'process.json'
New-Item -ItemType Directory -Path $formalOutput -Force | Out-Null
if (Test-Path -LiteralPath $formalPidFile) {
    $formalPrevious = Get-Content -LiteralPath $formalPidFile -Raw -Encoding utf8 | ConvertFrom-Json
    $formalExisting = Get-Process -Id $formalPrevious.pid -ErrorAction SilentlyContinue
    # PowerShell versions deserialize ISO dates as either DateTime or strings.
    # Compare UTC ticks so a live coordinator cannot be mistaken for a stale PID.
    if ($formalPrevious.started -is [datetime]) {
        $formalRecordedUtc = $formalPrevious.started.ToUniversalTime()
    } else {
        $formalRecordedUtc = [DateTimeOffset]::Parse([string]$formalPrevious.started, [Globalization.CultureInfo]::InvariantCulture).UtcDateTime
    }
    if ($formalExisting -and $formalExisting.ProcessName -eq 'python' -and $formalExisting.StartTime.ToUniversalTime().Ticks -eq $formalRecordedUtc.Ticks) {
        Write-Output "Training already running: $($formalPrevious.pid)"
        exit 0
    }
}
$env:PYTHONPATH = Join-Path $formalRoot 'src'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
$formalArgs = @('-X','utf8','-m','paper.formal.study','--config',(Join-Path $formalRoot 'experiments\formal_v3_study.json'),'--out',$formalOutput)
$formalProcess = Start-Process -FilePath $formalPython -ArgumentList $formalArgs -WorkingDirectory $formalRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $formalOutput 'coordinator.stdout.log') -RedirectStandardError (Join-Path $formalOutput 'coordinator.stderr.log') -PassThru
@{pid=$formalProcess.Id; started=$formalProcess.StartTime.ToUniversalTime().ToString('o'); python=$formalPython; output=$formalOutput} | ConvertTo-Json | Set-Content -LiteralPath $formalPidFile -Encoding utf8
Write-Output "Started formal study: PID $($formalProcess.Id)"
