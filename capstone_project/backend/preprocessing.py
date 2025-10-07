# src/nlp/preprocessing.py
import re
import unicodedata
from typing import Tuple, Optional

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# optional: use contractions lib if available
try:
    import contractions
    _HAS_CONTRACTIONS = True
except Exception:
    _HAS_CONTRACTIONS = False

# spaCy for lemmatization & stopwords
import spacy
# you must run: python -m spacy download en_core_web_sm
_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# --- Utility functions ----------------------------------------------------

def detect_language(text: str) -> str:
    """
    Detect language code for the text (e.g., 'en', 'hi').
    Returns 'unknown' on failure.
    """
    try:
        lang = detect(text)
        return lang
    except Exception:
        return "unknown"

def normalize_unicode(text: str) -> str:
    """Normalize unicode characters (NFKC) and remove BOM."""
    text = unicodedata.normalize("NFKC", text)
    # remove BOM and zero-width spaces
    text = text.replace("\ufeff", "").replace("\u200b", "")
    return text

def remove_urls_emails(text: str) -> str:
    """Remove URLs and email addresses."""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    return text

def remove_special_chars(text: str, keep_punct: bool = False) -> str:
    """
    Remove non-alphanumeric characters.
    If keep_punct=True we keep basic punctuation .,!?()-'
    """
    if keep_punct:
        text = re.sub(r"[^0-9a-zA-Z\s\.\,\!\?\-\(\)\'\/\:]", " ", text)
    else:
        text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    return text

def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def expand_contractions(text: str) -> str:
    """Expand contractions using the contractions library where available."""
    if _HAS_CONTRACTIONS:
        return contractions.fix(text)
    # fallback: small manual mapping (expand as needed)
    mapping = {
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "can't": "cannot",
        "won't": "will not",
        "i'm": "i am",
        "i've": "i have",
        "you're": "you are",
        "it's": "it is",
        "that's": "that is",
        "there's": "there is",
    }
    text_l = text.lower()
    for k, v in mapping.items():
        text_l = re.sub(r"\b" + re.escape(k) + r"\b", v, text_l)
    return text_l

# --- Main preprocessing functions -----------------------------------------

def clean_text_basic(text: str, keep_punct: bool = False) -> str:
    """
    Basic cleaning: unicode normalization, remove URLs/emails, expand contractions,
    remove special chars (or keep punctuation), lowercase, collapse whitespace.
    Returns cleaned string (not lemmatized).
    """
    if not text:
        return ""
    text = normalize_unicode(text)
    text = remove_urls_emails(text)
    text = expand_contractions(text)
    text = remove_special_chars(text, keep_punct=keep_punct)
    text = text.lower()
    text = collapse_whitespace(text)
    return text

def lemmatize_and_filter(text: str, remove_stopwords: bool = True) -> str:
    """
    Lemmatize and filter tokens: keep alphabetic tokens; remove stopwords if requested.
    Returns a space-joined lemmatized string suitable for embeddings / classification.
    """
    if not text:
        return ""
    doc = _nlp(text)
    tokens = []
    for token in doc:
        if not token.is_alpha:
            continue
        if remove_stopwords and token.is_stop:
            continue
        lemma = token.lemma_.lower().strip()
        if lemma:
            tokens.append(lemma)
    return " ".join(tokens)

def preprocess_for_retrieval(text: str, do_lemmatize: bool = True) -> Tuple[str, str]:
    """
    Full preprocessing convenience function.
    Returns tuple (cleaned_text_for_display, normalized_for_retrieval)
      - cleaned_text_for_display: readable cleaned form (good for LLM prompts/display)
      - normalized_for_retrieval: lemmatized, stopword-removed string for embeddings/FAISS
    """
    cleaned = clean_text_basic(text, keep_punct=False)
    if do_lemmatize:
        normalized = lemmatize_and_filter(cleaned, remove_stopwords=True)
    else:
        normalized = cleaned
    return cleaned, normalized

# Example small helper for numbers/dates normalization (optional)
def normalize_numbers(text: str) -> str:
    """Replace sequences of digits with a token 'NUM' — useful for some classifiers."""
    return re.sub(r"\d+", " NUM ", text)
