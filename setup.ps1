function Install-Dependencies {
    param (
        [string]$dir
    )
    Set-Location -Path $dir

    if (-not(Test-Path -Path "pyproject.toml")) {
        Write-Host "Directory $dir does not contain pyproject.toml!"
        exit 1
    }

    Write-Host "Installing dependencies in $dir..."
    poetry install
}

$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$subDirs = @(
    "eval_fusion\core"
)

Install-Dependencies -directory $rootDir

foreach ($subDir in $subDirs) {
    $dir = Join-Path -Path $rootDir -ChildPath $subDir

    if (-not (Test-Path -Path $dir)) {
        Write-Host "Error: Directory $dir does not exist!"
        exit 1
    }

    Install-Dependencies -directory $dir
}

Set-Location -Path $rootDir
