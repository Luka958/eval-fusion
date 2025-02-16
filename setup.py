import json
import os
import subprocess
import sys


def install_dependencies(dir: str, is_root: bool):
    pyproject_path = os.path.join(dir, 'pyproject.toml')

    if not os.path.isfile(pyproject_path):
        print(
            f'Error: Directory {dir} does not contain pyproject.toml!',
            file=sys.stderr,
        )
        sys.exit(1)

    if os.name == 'nt':
        if is_root:
            cmd = ['poetry', 'install', '--no-root', '--no-update']
        else:
            cmd = ['poetry', 'install', '--no-update']
    else:
        if is_root:
            cmd = ['poetry', 'install', '--no-root']
        else:
            cmd = ['poetry', 'install']

    target = 'root' if is_root else dir
    print(f'Installing dependencies in {target}...')

    try:
        subprocess.run(cmd, cwd=dir, check=True)

    except subprocess.CalledProcessError:
        print(f'Error: Poetry install failed in {dir}.', file=sys.stderr)
        sys.exit(1)


def init_vscode_settings(dir: str):
    cmd = ['poetry', 'env', 'info', '--path']
    env_path = subprocess.check_output(cmd, cwd=dir, encoding='utf-8').strip()

    vscode_dir = os.path.join(dir, '.vscode')
    os.makedirs(vscode_dir, exist_ok=True)
    settings_path = os.path.join(vscode_dir, 'settings.json')
    data = {'python.defaultInterpreterPath': env_path}

    with open(settings_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    print(f'Created VS Code settings at {settings_path}')


def init_vscode_workspace(root_dir: str, sub_dirs: list[str]):
    folders = [{'path': sub_dir} for sub_dir in sub_dirs]
    folders.append({'path': '.'})

    workspace_content = {
        'folders': folders,
        'settings': {'files.exclude': {'**/.vscode': True}},
    }
    workspace_file = os.path.join(root_dir, 'eval-fusion.code-workspace')

    with open(workspace_file, 'w', encoding='utf-8') as f:
        json.dump(workspace_content, f, indent=4)

    print(f'Created workspace file at {workspace_file}')


def main():
    vscode = len(sys.argv) > 1 and sys.argv[1].lower() == 'vscode'

    root_dir = os.path.dirname(os.path.realpath(__file__))
    install_dependencies(root_dir, is_root=True)

    if vscode:
        init_vscode_settings(root_dir)

    sub_dirs = [
        os.path.join('libs', 'core'),
        os.path.join('libs', 'community', 'deepeval'),
        os.path.join('libs', 'community', 'mlflow'),
        os.path.join('libs', 'community', 'phoenix'),
        os.path.join('libs', 'community', 'ragas'),
        os.path.join('libs', 'community', 'trulens'),
        os.path.join('libs', 'community', 'ragchecker'),
        os.path.join('libs', 'community', 'llama_index'),
        os.path.join('libs', 'vendor', 'openai'),
        os.path.join('libs', 'vendor', 'vertexai'),
        os.path.join('libs', 'test'),
    ]

    for sub_dir in sub_dirs:
        dir = os.path.join(root_dir, sub_dir)
        if not os.path.isdir(dir):
            print(f'Error: Directory {dir} does not exist!', file=sys.stderr)
            sys.exit(1)

        install_dependencies(dir, is_root=False)

        if vscode:
            init_vscode_settings(dir)

    os.chdir(root_dir)
    print('Poetry dependencies installation completed.')

    if vscode:
        init_vscode_workspace(root_dir, sub_dirs)


if __name__ == '__main__':
    main()
