# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Please run this script as Administrator"
    exit 1
}

# Check if Scoop is installed
if (-not (Get-Command scoop -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Scoop package manager..."
    # Set execution policy for current user only
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    
    # Install Scoop for current user
    $env:SCOOP='C:\Users\kanha\scoop'
    [Environment]::SetEnvironmentVariable('SCOOP', $env:SCOOP, 'User')
    $env:SCOOP_GLOBAL='C:\ProgramData\scoop'
    [Environment]::SetEnvironmentVariable('SCOOP_GLOBAL', $env:SCOOP_GLOBAL, 'Machine')
    $env:Path += ";$env:SCOOP\shims"
    
    # Download and run the Scoop installer
    Invoke-Expression (New-Object System.Net.WebClient).DownloadString('https://get.scoop.sh')
    
    # Add Scoop to PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# Check if Docker is installed
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker is not installed. Please install Docker Desktop for Windows from:"
    Write-Host "https://www.docker.com/products/docker-desktop/"
    exit 1
}

# Check if Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "Docker is not running. Please start Docker Desktop"
    exit 1
}

# Install act
if (-not (Get-Command act -ErrorAction SilentlyContinue)) {
    Write-Host "Installing act..."
    scoop install act
}

Write-Host "`nEnvironment setup complete! You can now run:"
Write-Host ".\scripts\test_workflows.ps1" 