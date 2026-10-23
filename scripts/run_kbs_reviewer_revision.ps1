param([string]$Python = 'python')
$ErrorActionPreference='Stop'
$projectRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:KBS_CPU_THREADS='2'
$env:USE_TF='0'
$env:USE_FLAX='0'
Push-Location $projectRoot
try {
  foreach ($script in @('src/experiments/kbs_calibration_fairness.py',
      'src/experiments/kbs_shift_features.py',
      'src/experiments/kbs_reviewer_revision.py',
      'src/experiments/kbs_verify_reviewer_revision.py',
      'src/experiments/kbs_verify_reviewer_provenance.py',
      'paper/kbs/build_reviewer_revision_results.py')) {
    if (Test-Path -LiteralPath 'experiments/kbs_strengthening/PAUSE') { break }
    & $Python -u $script
    if ($LASTEXITCODE -ne 0) { throw "Failed: $script. Checkpoints remain available for resume." }
  }
} finally { Pop-Location }
