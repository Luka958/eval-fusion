import os
import subprocess
import sys


def install_dependencies(directory: str, is_root: bool):
    pyproject_path = os.path.join(directory, 'pyproject.toml')

    if not os.path.isfile(pyproject_path):
        print(
            f'Error: Directory {directory} does not contain pyproject.toml!',
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

    target = 'root' if is_root else directory
    print(f'Installing dependencies in {target}...')

    try:
        subprocess.run(cmd, cwd=directory, check=True)

    except subprocess.CalledProcessError:
        print(f'Error: Poetry install failed in {directory}.', file=sys.stderr)
        sys.exit(1)


def main():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    install_dependencies(root_dir, is_root=True)

    sub_dirs = [
        os.path.join('libs', 'core'),
        os.path.join('libs', 'community', 'deepeval'),
        # os.path.join('libs', 'community', 'mlflow'),
        # os.path.join('libs', 'community', 'phoenix'),
        # os.path.join('libs', 'community', 'ragas'),
        # os.path.join('libs', 'community', 'trulens'),
        # os.path.join('libs', 'community', 'ragchecker'),
        # os.path.join('libs', 'community', 'llama_index'),
        # os.path.join('libs', 'vendor', 'openai'),
        # os.path.join('libs', 'vendor', 'vertexai'),
        # os.path.join('libs', 'test')
    ]

    for sub_dir in sub_dirs:
        full_dir = os.path.join(root_dir, sub_dir)
        if not os.path.isdir(full_dir):
            print(f'Error: Directory {full_dir} does not exist!', file=sys.stderr)
            sys.exit(1)

        install_dependencies(full_dir, is_root=False)

    os.chdir(root_dir)
    print('Poetry dependencies installation completed.')


if __name__ == '__main__':
    main()
