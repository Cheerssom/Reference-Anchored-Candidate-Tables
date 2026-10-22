param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:KBS_CPU_THREADS = '2'
$env:USE_TF = '0'
$env:USE_FLAX = '0'
$logDirectory = Join-Path $projectRoot 'experiments/kbs_independent_risk/logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
Push-Location $projectRoot
try {
  foreach ($encoder in @('l14','b32')) {
    foreach ($dataset in @('cifar','pets','imagenetr')) {
      if (Test-Path -LiteralPath 'experiments/kbs_strengthening/PAUSE') {
        Write-Output 'Pause is active; all completed risk checkpoints are retained.'
        exit 0
      }
      & $Python -u 'src/experiments/kbs_independent_risk.py' --encoder $encoder --dataset $dataset 2>&1 |
        Tee-Object -FilePath (Join-Path $logDirectory "$encoder-$dataset.log") -Append
      if ($LASTEXITCODE -ne 0) { throw "Risk run failed: $encoder / $dataset. Rerun this script to resume." }
    }
  }
} finally { Pop-Location }
