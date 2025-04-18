# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Please run this script as Administrator" -ForegroundColor Red
    exit 1
}

# Check if act is installed
if (-not (Get-Command act -ErrorAction SilentlyContinue)) {
    Write-Host "act is not installed. Please run setup_windows_env.ps1 first" -ForegroundColor Red
    exit 1
}

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "Docker is not running. Please start Docker Desktop" -ForegroundColor Red
    exit 1
}

# Function to run workflow with error handling
function Run-Workflow {
    param (
        [string]$WorkflowPath
    )
    
    Write-Host "`nTesting $WorkflowPath..." -ForegroundColor Cyan
    try {
        act -W $WorkflowPath
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Workflow test failed" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "Error running workflow: $_" -ForegroundColor Red
        exit 1
    }
}

# Run CI workflow
Run-Workflow -WorkflowPath ".github/workflows/ci.yml"

# Run CD workflow in dry-run mode
Write-Host "`nTesting CD workflow (dry run)..." -ForegroundColor Cyan
try {
    act -W .github/workflows/cd.yml -n
} catch {
    Write-Host "Error running CD workflow: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`nAll workflow tests completed successfully!" -ForegroundColor Green 