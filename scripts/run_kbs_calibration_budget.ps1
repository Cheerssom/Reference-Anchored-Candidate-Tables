param([string]$Python = 'python')
$ErrorActionPreference='Stop'
$projectRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:KBS_CPU_THREADS='2'
$env:USE_TF='0'
$env:USE_FLAX='0'
$logs=Join-Path $projectRoot 'experiments/kbs_calibration_budget/logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
Push-Location $projectRoot
try {
  & $Python -u 'src/experiments/kbs_calibration_pool.py' 2>&1 | Tee-Object -FilePath (Join-Path $logs 'features.log') -Append
  if ($LASTEXITCODE -ne 0) { throw 'Supplemental feature preparation failed; saved shards remain resumable.' }
  foreach ($encoder in @('l14','b32')) {
    foreach ($dataset in @('cifar','pets','imagenetr')) {
      if (Test-Path -LiteralPath 'experiments/kbs_strengthening/PAUSE') { exit 0 }
      & $Python -u 'src/experiments/kbs_calibration_budget.py' --encoder $encoder --dataset $dataset 2>&1 |
        Tee-Object -FilePath (Join-Path $logs "$encoder-$dataset.log") -Append
      if ($LASTEXITCODE -ne 0) { throw "Calibration-budget run failed: $encoder / $dataset. Rerun to resume." }
    }
  }
} finally { Pop-Location }
