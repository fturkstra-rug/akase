import boto3
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.auth import HTTPBasicAuth
import pandas as pd

# --- Config ---
MAX_WORKERS = 4   # Increased for I/O bound tasks
OUTPUT_FILE = "neighbors2.jsonl"
DEFAULT_SIZE = 10

# --- AWS Clients ---
secrets_client = boto3.client("secretsmanager", region_name="us-west-2")

# --- Fetch OpenSearch credentials ---
try:
    secret_response = secrets_client.get_secret_value(SecretId="OpenSearchCredentials")
    secret = json.loads(secret_response["SecretString"])
    OPENSEARCH_USER = secret["OPENSEARCH_USER"]
    OPENSEARCH_PASS = secret["OPENSEARCH_PASS"]
    OPENSEARCH_URL = secret["OPENSEARCH_URL"]
    OPENSEARCH_INDEX = secret["OPENSEARCH_INDEX"]
except Exception as e:
    print("❌ Error fetching secret:", e)
    exit(1)

# --- Prepare persistent session for OpenSearch ---
def create_session():
    session = requests.Session()
    session.auth = HTTPBasicAuth(OPENSEARCH_USER, OPENSEARCH_PASS)
    session.headers.update({"Content-Type": "application/json"})

    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# --- OpenSearch Query ---
def fetch_neighbors(session, doc):
    """Query OpenSearch for top N similar documents."""
    try:
        doc_id = doc.get("id")
        title = doc.get("title")

        if not doc_id or not title:
            return None

        search_query = {
            "size": DEFAULT_SIZE,
            "query": {
                "multi_match": {
                    "query": title,
                    "fields": ["title", "main_content"]
                }
            },
            "_source": ["id"]
        }

        response = session.post(
            f"{OPENSEARCH_URL}/{OPENSEARCH_INDEX}/_search",
            json=search_query,
            timeout=10
        )
        response.raise_for_status()
        hits = response.json().get("hits", {}).get("hits", [])
        neighbors = [h["_source"]["id"] for h in hits if h["_source"]["id"] != doc_id]

        return {"doc_id": doc_id, "neighbors": neighbors}

    except Exception as e:
        print(f"⚠️  Error querying OpenSearch for doc {doc.get('id')}: {e}")
        return None


# --- Main Logic ---
def main():
    start_time = time.time()

    # 1️⃣ List all S3 keys
    df = pd.read_json('docs.jsonl', lines=True)
    print(f"📄 Found {len(df)} documents.")

    # 3️⃣ Query OpenSearch concurrently with persistent session
    print("🔍 Querying OpenSearch for neighbors...")
    session = create_session()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        futures = {executor.submit(fetch_neighbors, session, doc): doc for doc in df.to_dict(orient="records")}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="doc"):
            result = future.result()
            if result:
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"✨ Done! Results saved to {OUTPUT_FILE}")
    print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
