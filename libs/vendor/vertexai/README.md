# eval-fusion-vertexai

## Setup
Create `google-cloud-credentials.json` and add its path to `GOOGLE_APPLICATION_CREDENTIALS`.
```json
{
  "type": "service_account",
  "project_id": "<project_id_>",
  "private_key_id": "<private_key_id>",
  "private_key": "<private_key>",
  "client_email": "<client_email>",
  "client_id": "<client_id>",
  "auth_uri": "<auth_uri>",
  "token_uri": "<token_uri>",
  "auth_provider_x509_cert_url": "<auth_provider_x509_cert_url>",
  "client_x509_cert_url": "<client_x509_cert_url>",
  "universe_domain": "<universe_domain>"
}
```


## Environment
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GOOGLE_CLOUD_AUTH_URL`
- `VERTEX_AI_PROJECT_ID`
- `VERTEX_AI_PROJECT_LOCATION`
- `VERTEX_AI_MODEL_LOCATION`