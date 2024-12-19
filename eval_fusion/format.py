import os
import sys
import subprocess


root_dir = os.getcwd()
dev_dir = os.path.join(root_dir, 'dev')

try:
    os.chdir(dev_dir)

    result = subprocess.run(
        args=['pipenv', 'run', 'black', '../'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print(result.stdout)
    print(result.stderr, file=sys.stderr)

except Exception as e:
    print(f'Error: {e}', file=sys.stderr)

finally:
    os.chdir(root_dir)
