from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
import pandas as pd
import nltk

for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)

def clean_html(text: str) -> str:
    # Remove HTML tags from text.
    return BeautifulSoup(text, 'html.parser').get_text(separator=' ')

def clean_html_fast(series: pd.Series) -> pd.Series:
    # OWI index only has minimal HTML tags (h1-6, p, ul/ol/li, pre, and a) so regex is sufficient (and faster).
    return (
        series
        .str.replace(r"<[^>]+>", " ", regex=True)  # strip tags
        .str.replace(r"\s+", " ", regex=True)  # collapse extra spaces
        .str.strip()
    )

# def sentence_tokenize(text: str) -> list[str]:
#     # Split text into sentences.
#     return sent_tokenize(text)

_tokenizer = PunktSentenceTokenizer() 

def sentence_tokenize(text: str) -> list[tuple[str, int, int]]:
    """
    Split text into sentences, returning (sentence, start_idx, end_idx).
    Example:
        "Hello world. Bye!" -> [("Hello world.", 0, 12), ("Bye!", 13, 17)]
    """
    spans = list(_tokenizer.span_tokenize(text))
    return [(text[start:end], start, end) for start, end in spans]

