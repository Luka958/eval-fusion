import vertexai

from decouple import config
from google.auth import default
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

from eval_fusion_test.utils.openai import OpenAILLM


GOOGLE_APPLICATION_CREDENTIALS = config('GOOGLE_APPLICATION_CREDENTIALS')
GOOGLE_CLOUD_AUTH_URL = config('GOOGLE_CLOUD_AUTH_URL')

VERTEX_AI_PROJECT_ID = config('VERTEX_AI_PROJECT_ID')
VERTEX_AI_PROJECT_LOCATION = config('VERTEX_AI_PROJECT_LOCATION')
VERTEX_AI_MODEL_LOCATION = config('VERTEX_AI_MODEL_LOCATION')
VERTEX_AI_BASE_URL = (
    f'https://{VERTEX_AI_MODEL_LOCATION}-aiplatform.googleapis.com/v1beta1/'
    f'projects/{VERTEX_AI_PROJECT_ID}/'
    f'locations/{VERTEX_AI_PROJECT_LOCATION}/'
    'endpoints/openapi'
)


class VertexAILLM(OpenAILLM):
    def __init__(self, model_name: str):
        credentials: Credentials = default(scopes=[GOOGLE_CLOUD_AUTH_URL])[0]
        auth_request = Request()
        credentials.refresh(auth_request)

        vertexai.init(project=VERTEX_AI_PROJECT_ID, location=VERTEX_AI_PROJECT_LOCATION)

        super().__init__(model_name, VERTEX_AI_BASE_URL, credentials.token)
