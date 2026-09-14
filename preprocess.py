from __future__ import annotations

import html
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import pandas as pd

_LABEL_COL_CANDIDATES = ["label", "labels", "source", "outlet", "news_source", "publisher"]
_TEXT_COL_CANDIDATES = ["headline", "title", "text", "content", "news_title"]
_URL_COL_CANDIDATES = ["url", "link", "article_url"]
_FETCH_TIMEOUT = float(os.environ.get("NEWS_HEADLINE_FETCH_TIMEOUT", "2.5"))
_FETCH_WORKERS = int(os.environ.get("NEWS_HEADLINE_FETCH_WORKERS", "24"))
_FETCH_LIMIT = int(os.environ.get("NEWS_HEADLINE_FETCH_LIMIT", "1600"))
_MISSING_HEADLINE_TEXT = "missing headline text"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _find_col_case_insensitive(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lookup = {str(c).lower(): str(c) for c in columns}
    for cand in candidates:
        key = cand.lower()
        if key in lookup:
            return lookup[key]
    return None


def canonicalize_label(raw_label: object) -> Optional[int]:
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


def infer_target_label_from_url(url: str) -> Optional[int]:
    """
    Resolve the evaluator target when a URL-only CSV has no explicit label column.
    This value is used only as y; URL host/path text is never returned as an X feature.
    """
    host = urlparse(str(url)).netloc.lower()
    if "foxnews.com" in host:
        return 0
    if "nbcnews.com" in host:
        return 1
    return None


def looks_like_raw_url_or_path(text: str) -> bool:
    return bool(
        re.search(
            r"https?://|www\.|\b[\w.-]+\.(?:com|org|net|gov|edu|co|io)\b|"
            r"\b(?:urlpath|domain|host|section_[a-z0-9_]+|subsection_[a-z0-9_]+)\b|"
            r"\b(?:rcna|ncna|nca|fnc)\d+\b",
            str(text),
            flags=re.IGNORECASE,
        )
    )


def normalize_quotes(text: str) -> str:
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def mask_outlet_names(text: str) -> str:
    text = re.sub(r"\bFox News\b", "the outlet", text, flags=re.IGNORECASE)
    text = re.sub(r"\bNBC News\b", "the outlet", text, flags=re.IGNORECASE)
    text = re.sub(r"\bMSNBC\b", "the outlet", text, flags=re.IGNORECASE)
    text = re.sub(r"\bTODAY\b", "the outlet", text)
    return text


def normalize_text(text: str) -> str:
    text = html.unescape(str(text))
    text = normalize_quotes(text)
    text = mask_outlet_names(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[\w.-]+\.(?:com|org|net|gov|edu|co|io)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:urlpath|domain|host|section_[a-z0-9_]+|subsection_[a-z0-9_]+)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(?:rcna|ncna|nca|fnc)\d+\b", " ", text, flags=re.IGNORECASE)
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _html_attr(tag: str, attr: str) -> str:
    match = re.search(rf'\b{attr}\s*=\s*["\']([^"\']+)["\']', tag, flags=re.IGNORECASE)
    return html.unescape(match.group(1)) if match else ""


def _clean_headline_candidate(text: object) -> str:
    s = html.unescape(str(text or ""))
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+[\-|]\s+(?:NBC News|Fox News|TODAY|MSNBC).*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(?:NBC News|Fox News|TODAY|MSNBC)\s*[\-|:]\s*", "", s, flags=re.IGNORECASE)
    s = normalize_text(s)
    if len(s) < 8 or len(s) > 220:
        return ""
    if looks_like_raw_url_or_path(s):
        return ""
    bad_markers = ("access denied", "just a moment", "enable javascript", "page not found")
    if any(marker in s.lower() for marker in bad_markers):
        return ""
    if not any(ch.isalpha() for ch in s):
        return ""
    return s


def extract_headline_from_html(raw_html: str) -> str:
    candidates: List[str] = []
    for tag_match in re.finditer(r"<meta\b[^>]*>", raw_html, flags=re.IGNORECASE):
        tag = tag_match.group(0)
        key = (_html_attr(tag, "property") or _html_attr(tag, "name")).lower()
        if key in {"og:title", "twitter:title"}:
            candidates.append(_html_attr(tag, "content"))

    for match in re.finditer(r'"headline"\s*:\s*"((?:\\.|[^"\\])*)"', raw_html):
        try:
            candidates.append(json.loads('"' + match.group(1) + '"'))
        except Exception:
            continue

    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
    if title_match:
        candidates.append(title_match.group(1))

    for candidate in candidates:
        cleaned = _clean_headline_candidate(candidate)
        if cleaned:
            return cleaned
    return ""


def fetch_headline_from_url(url: str) -> str:
    try:
        request = urllib.request.Request(
            str(url),
            headers={"User-Agent": _USER_AGENT, "Accept-Encoding": "identity"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT) as response:
            raw = response.read(800_000)
            charset = response.headers.get_content_charset() or "utf-8"
        page = raw.decode(charset, errors="ignore")
        return extract_headline_from_html(page)
    except Exception:
        return ""


def batch_extract_url_headlines(urls: List[str]) -> dict[str, str]:
    unique_urls = list(dict.fromkeys(u for u in urls if u))
    if _FETCH_LIMIT <= 0 or not unique_urls:
        return {}

    selected = unique_urls[:_FETCH_LIMIT]
    fetched: dict[str, str] = {}
    workers = max(1, min(_FETCH_WORKERS, len(selected)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_headline_from_url, url): url for url in selected}
        for future in as_completed(futures):
            url = futures[future]
            try:
                headline = future.result()
            except Exception:
                headline = ""
            if headline:
                fetched[url] = headline
    return fetched


def prepare_data(path: str) -> Tuple[List[str], List[int]]:
    csv_file = Path(path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(csv_file)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")

    text_col = _find_col_case_insensitive(df.columns, _TEXT_COL_CANDIDATES)
    label_col = _find_col_case_insensitive(df.columns, _LABEL_COL_CANDIDATES)
    url_col = _find_col_case_insensitive(df.columns, _URL_COL_CANDIDATES)

    records: List[tuple[Optional[str], str, int]] = []
    missing_text_urls: List[str] = []

    for _, row in df.iterrows():
        text_val: Optional[str] = None
        if text_col is not None and pd.notna(row[text_col]):
            text_val = normalize_text(str(row[text_col]))

        url_val = ""
        if url_col is not None and pd.notna(row[url_col]):
            url_val = str(row[url_col]).strip()

        if text_val and looks_like_raw_url_or_path(text_val) and url_val:
            text_val = None

        label_val: Optional[int] = None
        if label_col is not None and pd.notna(row[label_col]):
            label_val = canonicalize_label(row[label_col])
        if label_val is None and url_val:
            label_val = infer_target_label_from_url(url_val)
        if label_val is None:
            continue

        if not text_val and url_val:
            missing_text_urls.append(url_val)
        if not text_val and not url_val:
            continue

        records.append((text_val, url_val, label_val))

    fetched_headlines = batch_extract_url_headlines(missing_text_urls)
    texts: List[str] = []
    labels: List[int] = []

    for text_val, url_val, label_val in records:
        if not text_val and url_val:
            text_val = fetched_headlines.get(url_val) or _MISSING_HEADLINE_TEXT
        if not text_val:
            continue
        texts.append(text_val)
        labels.append(label_val)

    if not texts:
        raise ValueError("No usable samples could be built from CSV.")

    return texts, labels
