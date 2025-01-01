import vertexai

from decouple import config
from eval_fusion_core.base import EvalFusionBaseLLM
from google.auth import default
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from openai import AsyncOpenAI, OpenAI


GOOGLE_APPLICATION_CREDENTIALS = config('GOOGLE_APPLICATION_CREDENTIALS')
GOOGLE_CLOUD_AUTH_URL = config('GOOGLE_CLOUD_AUTH_URL')

VERTEX_AI_PROJECT_ID = config('VERTEX_AI_PROJECT_ID')
VERTEX_AI_PROJECT_LOCATION = config('VERTEX_AI_PROJECT_LOCATION')
VERTEX_AI_MODEL_LOCATION = config('VERTEX_AI_MODEL_LOCATION')

BASE_URL = (
    f'https://{VERTEX_AI_MODEL_LOCATION}-aiplatform.googleapis.com/v1beta1/'
    f'projects/{VERTEX_AI_PROJECT_ID}/'
    f'locations/{VERTEX_AI_PROJECT_LOCATION}/'
    'endpoints/openapi'
)


class VertexAILLM(EvalFusionBaseLLM):
    def __init__(self, model_id: str):
        self.model_id = model_id

        credentials: Credentials = default(scopes=[GOOGLE_CLOUD_AUTH_URL])[0]
        auth_request = Request()
        credentials.refresh(auth_request)

        vertexai.init(project=VERTEX_AI_PROJECT_ID, location=VERTEX_AI_PROJECT_LOCATION)

        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=credentials.token,
        )
        self.async_client = AsyncOpenAI(
            base_url=BASE_URL,
            api_key=credentials.token,
        )

    def get_name(self) -> str:
        return self.model_id

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt},
            ],
        )

        return completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        completion = await self.async_client.chat.completions.create(
            model=self.model_id,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt},
            ],
        )

        return completion.choices[0].message.content
