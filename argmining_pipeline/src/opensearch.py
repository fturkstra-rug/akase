import requests
import time
import pandas as pd
import json
import logging
import boto3
logger = logging.getLogger(__name__)


# ==== CONFIG ====
secrets_client = boto3.client("secretsmanager", region_name="us-west-2")

try:
    secret_response = secrets_client.get_secret_value(SecretId="OpenSearchCredentials")
    secret = json.loads(secret_response["SecretString"])
    USERNAME = secret["OPENSEARCH_USER"]
    PASSWORD = secret["OPENSEARCH_PASS"]
    OPENSEARCH_URL = secret["OPENSEARCH_URL"]
    INDEX_NAME = secret["OPENSEARCH_INDEX"]
except Exception as e:
    print("❌ Error fetching secret:", e)
    exit(1)

CHUNK_SIZE = 2000  # number of lines per bulk request (~1000 docs)
MAX_RETRIES = 3
MAX_CHARS_MAIN_CONTENT = 2000  # None = no truncation

SELECTED_FIELDS = [
    'id',
    'title',
    'main_content',
    'url',
    'url_domain',
    'warc_date'
]

def prepare_payload(df: pd.DataFrame) -> list[dict]:
    payload = []

    for _, row in df.iterrows():
        doc_id = row['id']

        if not row['valid']:
            logger.warning(f'Skipping invalid record: {doc_id}')
            continue

        action = {'index': {'_index': 'web_index', '_id': doc_id}}

        index_doc = {}
        for field in SELECTED_FIELDS:
            value = row.get(field, None)
            if field == 'main_content' and isinstance(value, str) and MAX_CHARS_MAIN_CONTENT:
                value = value[:MAX_CHARS_MAIN_CONTENT]
            index_doc[field] = value

        payload.append(action)
        payload.append(index_doc)
    
    return payload

def upload_chunk(chunk, retries=0):
    url = f'{OPENSEARCH_URL}/{INDEX_NAME}/_bulk'
    headers = {'Content-Type': 'application/x-ndjson'}

    response = requests.post(
        url,
        headers=headers,
        data=chunk.encode('utf-8'),
        auth=(USERNAME, PASSWORD)
    )

    if response.status_code == 200:
        # Check if some documents failed to upload
        response_json = response.json()
        if response_json.get('errors'):
            print('Some documents failed:', response_json)
            return False
        return True

    elif response.status_code in (429, 500, 503) and retries < MAX_RETRIES:
        wait = 2 ** retries
        print(f'Retry {retries+1}/{MAX_RETRIES} after {wait}s (status {response.status_code})')
        time.sleep(wait)
        return upload_chunk(chunk, retries + 1)

    else:
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            print('Bulk upload failed:', e, response.text)
        return False
    
def upload_to_index(df: pd.DataFrame) -> None:

    # Quick scheme fix (ows_index and ows_genai should be boolean but are ('1.0'/'0.0'))
    # df['ows_index'] = df['ows_index'].apply(lambda x: bool(int(x)))
    # df['ows_genai'] = df['ows_genai'].apply(lambda x: bool(int(x)))

    payload = prepare_payload(df)

    buffer = []
    for i, row in enumerate(payload, 1):
        buffer.append(json.dumps(row))

        if i % CHUNK_SIZE == 0:
            chunk_data = '\n'.join(buffer) + '\n'
            upload_chunk(chunk_data)
            buffer = []

    if buffer: # leftover docs
        chunk_data = '\n'.join(buffer) + '\n'
        upload_chunk(chunk_data)
    