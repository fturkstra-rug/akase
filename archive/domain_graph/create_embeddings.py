import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Cache the model 
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("intfloat/e5-large-v2")
    return _model

def batched_encode(model, texts, batch_size=1, device="cuda"):
    """Encodes text in small batches and returns stacked embeddings on CPU."""
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            emb = model.encode(batch, convert_to_tensor=True, device=device)
            embeddings.append(emb.cpu())  # Move to CPU immediately to free GPU

    return torch.cat(embeddings, dim=0)  # [num_texts, emb_dim]

def embed(texts):
    model = get_model()
    return batched_encode(model, texts, batch_size=1, device='cuda')
