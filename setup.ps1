function Install-Dependencies {
    param (
        [string]$dir,
        [bool]$isRoot
    )
    Set-Location -Path $dir

    if (-not(Test-Path -Path "pyproject.toml")) {
        Write-Host "Directory $dir does not contain pyproject.toml!"
        exit 1
    }

    if ($isRoot) {
        Write-Host "Installing dependencies in root..."
        poetry install
    }
    else {
        Write-Host "Installing dependencies in $dir..."
        poetry install --no-root
    }
}

$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$subDirs = @(
    "eval_fusion\core"
)

Install-Dependencies -dir $rootDir -isRoot $true

foreach ($subDir in $subDirs) {
    $dir = Join-Path -Path $rootDir -ChildPath $subDir

    if (-not (Test-Path -Path $dir)) {
        Write-Host "Error: Directory $dir does not exist!"
        exit 1
    }

    Install-Dependencies -dir $dir -isRoot $false
}

Set-Location -Path $rootDir
