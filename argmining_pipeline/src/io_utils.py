import pandas as pd
from collections.abc import Iterator
import pickle
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def load_parquet_files(dir: str | Path) -> Iterator[pd.DataFrame]:
    dir = Path(dir).expanduser()
    for fpath in dir.rglob("metadata_*.parquet"):
        yield pd.read_parquet(fpath)

def load_pickle(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def save_json(obj: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_json(path: str | Path) -> dict:
    path = Path(path)
    with open(path, 'r') as f:
        return json.load(f)

def upload_to_s3(client, doc: dict, bucket: str, prefix: str):
    key = f"{prefix}/{doc['id']}.json"
    client.put_object(Bucket=bucket, Key=key, Body=json.dumps(doc, default=str).encode('utf-8'))
    return doc['id']

def bulk_upload_to_s3(client, df: pd.DataFrame, bucket: str, prefix: str, max_workers: int = 20):
    docs = df.to_dict(orient="records")
    uploader = partial(upload_to_s3, client, bucket=bucket, prefix=prefix)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        doc_ids = list(executor.map(uploader, docs))

    return doc_ids

def save_parquet(df: pd.DataFrame, path: str): ...
