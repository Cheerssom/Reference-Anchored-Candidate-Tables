param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$env:USE_TF = '0'
$env:USE_FLAX = '0'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
 & $Python src/experiments/kbs_full_fusion_retuning.py
 if ($LASTEXITCODE -ne 0) { throw 'Full fusion retuning failed.' }
 if (Test-Path experiments/kbs_strengthening/PAUSE) { return }
 & $Python src/experiments/kbs_verify_full_fusion_retuning.py
 if ($LASTEXITCODE -ne 0) { throw 'Retuning verification failed.' }
 & $Python paper/kbs/build_full_fusion_retuning.py
 if ($LASTEXITCODE -ne 0) { throw 'Retuning summary failed.' }
} finally { Pop-Location }
