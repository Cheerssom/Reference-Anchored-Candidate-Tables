param([string]$Python = 'python')
$ErrorActionPreference='Stop'
$projectRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:KBS_CPU_THREADS='2'
$env:USE_TF='0'
$env:USE_FLAX='0'
Push-Location $projectRoot
try {
  foreach ($script in @('src/experiments/kbs_mechanism_followup.py',
      'src/experiments/test_kbs_mechanism_followup.py',
      'paper/kbs/build_mechanism_followup.py')) {
    if (Test-Path -LiteralPath 'experiments/kbs_strengthening/PAUSE') { break }
    & $Python -u $script
    if ($LASTEXITCODE -ne 0) { throw "Failed: $script. Completed checkpoints are preserved." }
  }
} finally { Pop-Location }
