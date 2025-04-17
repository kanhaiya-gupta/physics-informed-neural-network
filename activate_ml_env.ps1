# Script to activate the ml_env Conda environment

# Check if Conda command is available
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Conda command not found. Please ensure Conda is installed and in your PATH"
    exit 1
}

# Get Conda base path
$CONDA_PATH = conda info --base

Write-Host "Found Conda installation at: $CONDA_PATH"

# Activate the ml_env environment
Write-Host "Activating ml_env environment..."
conda activate ml_env

# Verify activation
$currentEnv = conda env list | Where-Object { $_ -match '\*' } | ForEach-Object { $_.Split()[0] }
if ($currentEnv -ne "ml_env") {
    Write-Host "Error: Failed to activate ml_env environment"
    Write-Host "Current environment: $currentEnv"
    exit 1
} else {
    Write-Host "Successfully activated ml_env environment"
} 