param(
  [ValidateSet('All','Download','Features','Experiments','Depth','ValidationBudget','Status','Pause','Resume')]
  [string]$Stage = 'All',
  [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$controlDir = Join-Path $projectRoot 'experiments/kbs_strengthening'
New-Item -ItemType Directory -Force $controlDir | Out-Null
$pausePath = Join-Path $controlDir 'PAUSE'
if ($Stage -eq 'Pause') {
  Set-Content -LiteralPath $pausePath -Value 'Pause requested by the local operator.'
  Write-Output 'Pause requested. The running job will stop at its next safe checkpoint.'
  exit 0
}
if ($Stage -eq 'Status') {
  Write-Output ('Pause requested: ' + (Test-Path -LiteralPath $pausePath))
  Get-ChildItem -LiteralPath $controlDir -Filter complete.json -Recurse |
    ForEach-Object { Write-Output $_.FullName }
  $depthDir = Join-Path $projectRoot 'experiments/kbs_depth'
  if (Test-Path -LiteralPath $depthDir) {
    Write-Output ('Depth configurations complete: ' + @(Get-ChildItem -LiteralPath $depthDir -Filter complete.json -Recurse).Count + ' / 90')
  }
  $budgetDir = Join-Path $projectRoot 'experiments/kbs_validation_budget'
  if (Test-Path -LiteralPath $budgetDir) {
    Write-Output ('Validation-budget datasets complete: ' + @(Get-ChildItem -LiteralPath $budgetDir -Filter complete.json -Recurse).Count + ' / 3')
  }
  Get-ChildItem (Join-Path $projectRoot 'experiments/kbs_imagenetr') -Filter 'frozen_imagenetr.pt' -ErrorAction SilentlyContinue |
    Select-Object Name,Length,LastWriteTime
  exit 0
}
if ($Stage -eq 'Resume') {
  if (Test-Path -LiteralPath $pausePath) { Remove-Item -LiteralPath $pausePath }
  $Stage = 'All'
}
if (Test-Path -LiteralPath $pausePath) {
  throw 'A pause is active. Use -Stage Resume to clear it and continue.'
}
$logDir = Join-Path $controlDir 'logs'
$env:USE_TF = '0'
$env:USE_FLAX = '0'
New-Item -ItemType Directory -Force $logDir | Out-Null
function Run-SavedStage([string]$Script,[string[]]$Arguments,[string]$Log) {
  if (Test-Path -LiteralPath $pausePath) { exit 0 }
  Push-Location $projectRoot
  try {
    & $Python -u $Script @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $logDir $Log) -Append
    if ($LASTEXITCODE -ne 0) { throw "Stage failed; progress was retained. Inspect $Log and rerun to resume." }
  } finally { Pop-Location }
}
if ($Stage -in @('All','Download')) {
  if (-not (Test-Path (Join-Path $projectRoot 'experiments/kbs_imagenetr/frozen_imagenetr.pt'))) {
    Run-SavedStage 'src/experiments/kbs_download_imagenetr.py' @() 'imagenetr-download.log'
  }
}
if ($Stage -in @('All','Features')) {
  Run-SavedStage 'src/experiments/kbs_imagenetr_data.py' @() 'imagenetr-features.log'
}
if ($Stage -in @('All','Experiments')) {
  foreach ($dataset in @('cifar','pets','imagenetr')) {
    Run-SavedStage 'src/experiments/kbs_strengthen.py' @('--dataset',$dataset) "$dataset-all.log"
    Run-SavedStage 'src/experiments/kbs_fenec_euclidean.py' @('--dataset',$dataset) "$dataset-euclidean.log"
    Run-SavedStage 'src/experiments/kbs_fenec_vit.py' @('--dataset',$dataset) "$dataset-fenec-vit.log"
  }
}
if ($Stage -in @('All','Depth')) {
  foreach ($dataset in @('cifar','pets','imagenetr')) {
    Run-SavedStage 'src/experiments/kbs_depth.py' @('--dataset',$dataset) "$dataset-depth.log"
  }
}
if ($Stage -in @('All','ValidationBudget')) {
  Run-SavedStage 'src/experiments/kbs_validation_budget.py' @() 'validation-budget.log'
}
