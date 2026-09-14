from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import pandas as pd


_LABEL_COL_CANDIDATES = ["label", "labels", "source", "outlet", "news_source", "publisher"]
_TEXT_COL_CANDIDATES = ["headline", "title", "text", "content", "news_title"]
_URL_COL_CANDIDATES = ["url", "link", "article_url"]


@dataclass
class PreparedData:
    """Container for preprocessed texts and labels."""

    texts: List[str]
    labels: List[int]


def _find_col_case_insensitive(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Case-insensitive column lookup; return the first matching candidate."""
    lookup = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def canonicalize_label(raw_label: object) -> Optional[int]:
    """Normalize various label spellings to binary ids: Fox=0, NBC=1."""
    if raw_label is None:
        return None
    text = str(raw_label).strip().lower()
    if text == "":
        return None

    if text in {"0", "fox", "foxnews", "fox news"}:
        return 0
    if text in {"1", "nbc", "nbcnews", "nbc news"}:
        return 1
    return None


def infer_label_from_url(url: str) -> Optional[int]:
    """Infer label from URL host when explicit labels are missing."""
    host = urlparse(str(url)).netloc.lower()
    if "foxnews.com" in host:
        return 0
    if "nbcnews.com" in host:
        return 1
    return None


def url_to_pseudo_headline(url: str) -> str:
    """
    Build pseudo-headline text from URL path when real headline text is absent.
    Note: this is a fallback path and should not be used for final headline-first training.
    """
    parsed = urlparse(str(url))
    path = unquote(parsed.path or "")
    path = path.replace(".print", " ")
    path = path.replace(".html", " ")
    path = path.replace("/", " ")
    path = path.replace("-", " ")
    path = path.replace("_", " ")
    path = html.unescape(path)
    path = re.sub(r"\s+", " ", path).strip()
    return path


def normalize_text(text: str) -> str:
    """Apply minimal text normalization to reduce noisy characters."""
    text = html.unescape(str(text))
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_dataset_from_csv(
    csv_path: str,
    *,
    allow_url_fallback: bool = True,
    require_text_column: bool = False,
    remove_duplicate_urls: bool = False,
    remove_duplicate_headlines: bool = False,
    min_headline_chars: int = 0,
    drop_symbol_only_headlines: bool = False,
) -> PreparedData:
    """
    Build training/evaluation samples from CSV.
    Rules:
    1) Prefer headline/title-like text columns.
    2) If allow_url_fallback=True, generate pseudo-headlines from URL when text is missing.
    3) Prefer explicit label/source columns; otherwise infer label from URL host.
    4) Optional dataset cleaning controls:
       - remove duplicate URLs
       - remove duplicate headlines
       - drop short headlines
       - drop symbol-only headlines
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_file)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    text_col = _find_col_case_insensitive(df.columns, _TEXT_COL_CANDIDATES)
    label_col = _find_col_case_insensitive(df.columns, _LABEL_COL_CANDIDATES)
    url_col = _find_col_case_insensitive(df.columns, _URL_COL_CANDIDATES)

    if require_text_column and text_col is None:
        raise ValueError(
            "CSV does not contain a headline/title/text-style column. "
            "If this is intentional, disable require_text_column or enable allow_url_fallback."
        )

    texts: List[str] = []
    labels: List[int] = []
    seen_urls: set[str] = set()
    seen_headlines: set[str] = set()

    for _, row in df.iterrows():
        # Resolve text first
        text_val: Optional[str] = None
        if text_col is not None and pd.notna(row[text_col]):
            text_val = normalize_text(str(row[text_col]))

        url_val = ""
        if url_col is not None and pd.notna(row[url_col]):
            url_val = str(row[url_col]).strip()
        elif "url" in df.columns and pd.notna(row["url"]):
            url_val = str(row["url"]).strip()

        if not text_val:
            if allow_url_fallback and url_val:
                text_val = normalize_text(url_to_pseudo_headline(url_val))
            else:
                continue

        if remove_duplicate_urls and url_val:
            url_key = url_val.strip().lower()
            if url_key in seen_urls:
                continue
            seen_urls.add(url_key)

        # Resolve label next
        label_val: Optional[int] = None
        if label_col is not None and pd.notna(row[label_col]):
            label_val = canonicalize_label(row[label_col])

        if label_val is None and url_val:
            label_val = infer_label_from_url(url_val)

        # Skip rows with unresolved labels to keep X/y aligned
        if label_val is None:
            continue

        if text_val == "":
            continue

        if min_headline_chars > 0 and len(text_val) < min_headline_chars:
            continue

        if drop_symbol_only_headlines:
            if not any(ch.isalnum() for ch in text_val):
                continue

        if remove_duplicate_headlines:
            headline_key = text_val.strip().lower()
            if headline_key in seen_headlines:
                continue

        texts.append(text_val)
        labels.append(label_val)
        if remove_duplicate_headlines:
            seen_headlines.add(text_val.strip().lower())

    if not texts:
        raise ValueError(
            "No usable samples could be built from CSV. "
            "Check whether columns include headline/title/url and label/source."
        )

    return PreparedData(texts=texts, labels=labels)
