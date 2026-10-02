param(
    [Parameter(Mandatory = $true)][int]$TrainingPid,
    [string]$Python = "D:\Anaconda3\python.exe",
    [string]$ProjectRoot = "E:\Python-program\Detect\ultralytics",
    [string]$RunName = "telecom_yolo12n_p2_fault_finetune_pycharm"
)

$ErrorActionPreference = "Stop"
$runDir = Join-Path $ProjectRoot "runs\train\$RunName"
$weights = Join-Path $runDir "weights\best.pt"
$log = Join-Path $runDir "test_evaluation.log"

try {
    Wait-Process -Id $TrainingPid
} catch {
    Add-Content -LiteralPath $log -Value "$(Get-Date -Format s) Training process ended before the evaluator attached."
}

if (-not (Test-Path -LiteralPath $weights)) {
    Add-Content -LiteralPath $log -Value "$(Get-Date -Format s) Evaluation skipped because best.pt was not created."
    exit 1
}

& $Python "val.py" --model $weights --split test --imgsz 1280 --batch 4 *>> $log
exit $LASTEXITCODE
