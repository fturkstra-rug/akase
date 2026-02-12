import boto3
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.auth import HTTPBasicAuth

# --- Config ---
FOLDER = "web_docs/"
BUCKET_NAME = "akasearch-deep-storage"
MAX_WORKERS = 16   # Increased for I/O bound tasks
OUTPUT_FILE = "docs.jsonl"
DEFAULT_SIZE = 10

# --- AWS Clients ---
s3 = boto3.client("s3")

# --- S3 Helpers ---
def list_keys():
    """List all object keys under the given folder."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=FOLDER):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):  # skip folder markers
                yield key


def fetch_file(key: str):
    """Download one JSON file from S3."""
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        content = response["Body"].read().decode("utf-8")
        return json.loads(content)
    except Exception as e:
        print(f"⚠️  Error fetching {key}: {e}")
        return None

# --- Main Logic ---
def main():
    start_time = time.time()

    # 1️⃣ List all S3 keys
    keys = list(list_keys())
    print(f"📄 Found {len(keys)} documents in {BUCKET_NAME}/{FOLDER}")

    # 2️⃣ Download documents from S3 concurrently
    print("⬇️  Downloading S3 documents...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        docs = list(filter(None, tqdm(
            executor.map(fetch_file, keys),
            total=len(keys),
            desc="Fetching from S3",
            unit="file"
        )))

    print(f"✅ Fetched {len(docs)} valid documents from S3")

    print(f"💾 Saving documents to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for doc in docs:
            f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"✨ Done! Results saved to {OUTPUT_FILE}")
    print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
