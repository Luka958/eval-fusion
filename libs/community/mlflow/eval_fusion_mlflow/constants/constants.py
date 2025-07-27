import os


EXPERIMENT_NAME = 'eval_fusion_experiment'
MODEL_NAME = 'custom_llm'
ENDPOINT_NAME = 'chat'
MODEL = f'endpoints:/{ENDPOINT_NAME}'
ARTIFACT_KEY_SETTINGS = 'settings'

MODELS_HOST = '127.0.0.1'
MODELS_PORT = 5030
MODELS_ENV_MANAGER = 'local'

DEPLOYMENTS_HOST = '127.0.0.1'
DEPLOYMENTS_PORT = 5031


DEPLOYMENTS_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, 'config.yaml')
)
