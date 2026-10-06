param([Parameter(Mandatory=$true)][string]$Snapshot)
$ErrorActionPreference = 'Stop'
if ($Snapshot -notmatch '^\d{8}_\d{6}$') { throw 'Invalid snapshot identifier' }
$deliveryRoot = 'F:\动态因果\results\formal_v3\delivery'
$archiveRoot = 'E:\智能交通\论文20260111_实验资料包_20260920\10_真实证据与正式训练_v3'
$snapshotSource = [System.IO.Path]::GetFullPath((Join-Path $deliveryRoot $Snapshot))
$snapshotTarget = [System.IO.Path]::GetFullPath((Join-Path $archiveRoot $Snapshot))
if (-not $snapshotSource.StartsWith($deliveryRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Source outside delivery root' }
if (-not $snapshotTarget.StartsWith($archiveRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Target outside archive root' }
$verification = Get-Content -LiteralPath (Join-Path $snapshotSource 'verification.json') -Raw -Encoding utf8 | ConvertFrom-Json
if ([System.IO.Path]::GetFileName($verification.archive) -ne $verification.archive) { throw 'Invalid archive name' }
$snapshotZip = Join-Path $snapshotSource $verification.archive
if ((Get-FileHash -LiteralPath $snapshotZip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $verification.archive_sha256) { throw 'Source archive hash mismatch' }
New-Item -ItemType Directory -Path $snapshotTarget -Force | Out-Null
foreach ($snapshotFile in @($verification.archive, 'verification.json', '交付说明.md')) {
    $copySource = Join-Path $snapshotSource $snapshotFile
    $copyTarget = Join-Path $snapshotTarget $snapshotFile
    $sourceHash = (Get-FileHash -LiteralPath $copySource -Algorithm SHA256).Hash
    if (Test-Path -LiteralPath $copyTarget) {
        if ((Get-FileHash -LiteralPath $copyTarget -Algorithm SHA256).Hash -ne $sourceHash) { throw 'Existing destination differs; no overwrite performed' }
    } else {
        Copy-Item -LiteralPath $copySource -Destination $copyTarget
    }
    if ((Get-FileHash -LiteralPath $copyTarget -Algorithm SHA256).Hash -ne $sourceHash) { throw 'Destination hash mismatch' }
}
Write-Output "Verified snapshot delivered: $snapshotTarget"
