from collections import Counter
from typing import List, Tuple
import glob
import json
from pathlib import Path
import numpy as np


def ngram_overlap(s1: str, s2: str, n: int = 2) -> Tuple[float, float, float]:
    """
    Computes n-gram overlap (precision, recall, F1) between two sentences.
    
    Args:
        s1 (str): First sentence (e.g., predicted).
        s2 (str): Second sentence (e.g., reference).
        n (int): Size of n-grams.
    
    Returns:
        Tuple[float, float, float]: (precision, recall, f1)
    """
    def get_ngrams(text: str, n: int) -> List[Tuple[str]]:
        tokens = text.lower().split()
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    ngrams1 = Counter(get_ngrams(s1, n))
    ngrams2 = Counter(get_ngrams(s2, n))

    intersection = sum((ngrams1 & ngrams2).values())
    total_pred = sum(ngrams1.values())
    total_ref = sum(ngrams2.values())

    precision = intersection / total_pred if total_pred else 0
    recall = intersection / total_ref if total_ref else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return precision, recall, f1


directory = Path("../debates")
files = glob.glob(str(directory / "*.json"))
f1_scores = []

for i, file in enumerate(files):
    with open(file, 'r') as f:
        data = json.load(f)
    
    precision, recall, f1 = ngram_overlap(data['rebuttal_1'], data['rebuttal_2'], n=2)
    print(i, f"{precision:.2f}", f"{recall:.2f}", f"{f1:.2f}")
    f1_scores.append(f1)

print(np.mean(f1_scores))
