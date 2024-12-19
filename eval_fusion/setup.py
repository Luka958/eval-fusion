import subprocess
from setuptools import setup, Command


INFRASTRUCTURE_DIRECTORIES = [
    #'infrastructure/eval/bench',
    'infrastructure/eval/deepeval',
    'infrastructure/eval/mlflow',
    'infrastructure/eval/phoenix',
    'infrastructure/eval/ragas',
    'infrastructure/eval/trulens'
]  # TODO


class SetupInfrastructureCommand(Command):  # TODO for dev and core
    description = 'Setup pipenv environments in infrastructure subdirectories'

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for dir in INFRASTRUCTURE_DIRECTORIES:
            print(f'\nSetting up pipenv environment in {dir}...')
            try:
                subprocess.check_call(['pipenv', 'install'], cwd=dir)
                print(f'Environment set up successfully in {dir}')

            except subprocess.CalledProcessError as e:
                print(f'Failed to set up environment in {dir}: {e}')


setup(
    name='eval_fusion',
    version='0.1',
    packages=[],
    cmdclass={
        'setup_infrastructure': SetupInfrastructureCommand,
    },
)