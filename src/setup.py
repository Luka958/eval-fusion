import os
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

def create_virtualenv(path: Path) -> NoReturn:
    try:
        subprocess.check_call([sys.executable, '-m', 'venv', str(path)])
        print(f'✅ Virtual environment created at {path}')
    except subprocess.CalledProcessError:
        print(f'❌ Failed to create virtual environment at {path}')

def install_requirements(env_path: Path, requirements_file: Path) -> NoReturn:
    pip_path = env_path / 'bin' / 'pip' if os.name != 'nt' else env_path / 'Scripts' / 'pip.exe'
    try:
        subprocess.check_call([str(pip_path), 'install', '-r', str(requirements_file)])
        print(f'✅ Installed requirements from {requirements_file}')
    except subprocess.CalledProcessError:
        print(f'❌ Failed to install requirements from {requirements_file}')

def setup_environments(root_dir: Path) -> NoReturn:
    for subproject in root_dir.iterdir():
        requirements_file = subproject / 'requirements.txt'
        env_dir = subproject / 'env'
        
        if requirements_file.is_file():
            print(f'\n🔄 Setting up environment for {subproject.name}')
            create_virtualenv(env_dir)
            install_requirements(env_dir, requirements_file)
        else:
            print(f'⚠️ No requirements.txt found in {subproject}. Skipping...')

if __name__ == '__main__':
    root_dir = Path(__file__).parent
    print('🚀 Starting environment setup for all subprojects...\n')
    setup_environments(root_dir)
    print('\n🎉 Environment setup complete for all subprojects!')
