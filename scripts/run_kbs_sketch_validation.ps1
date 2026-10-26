param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$env:USE_TF = '0'
$env:USE_FLAX = '0'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
 & $Python src/experiments/kbs_sketch_pipeline.py
 if ($LASTEXITCODE -ne 0) { throw 'Sketch validation stopped. Completed checkpoints are preserved.' }
} finally { Pop-Location }
