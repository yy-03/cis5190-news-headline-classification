from __future__ import annotations

import html
import io
import re
from pathlib import Path
from typing import Any, Iterable, List
from urllib.parse import unquote, urlparse

import joblib
import torch
from torch import nn

_MISSING_HEADLINE_TEXT = "missing headline text"


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


class Model(nn.Module):
    """
    KK submission model wrapper.
    It uses the original char_wb TF-IDF + LogisticRegression pipeline, packed
    into model.pt so the HF uploader can use the standard three-file format.
    """

    def __init__(self, weights_path: str | None = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("artifact_bytes", torch.zeros((1,), dtype=torch.uint8))
        self.register_buffer("artifact_len", torch.zeros((1,), dtype=torch.int64))
        self.pipeline = None
        self.decision_threshold = None
        self.decision_threshold_kind = None
        self._load_pipeline(weights_path)

    def load_state_dict(self, state_dict, strict: bool = True):
        payload = state_dict.get("artifact_bytes")
        if isinstance(payload, torch.Tensor) and tuple(payload.shape) != tuple(self.artifact_bytes.shape):
            self._buffers["artifact_bytes"] = torch.zeros_like(payload, dtype=torch.uint8)
        artifact_len = state_dict.get("artifact_len")
        if isinstance(artifact_len, torch.Tensor) and tuple(artifact_len.shape) != tuple(self.artifact_len.shape):
            self._buffers["artifact_len"] = torch.zeros_like(artifact_len, dtype=torch.int64)
        result = super().load_state_dict(state_dict, strict=strict)
        self._pipeline_from_buffers()
        return result

    @staticmethod
    def _default_artifact_path() -> Path:
        root = Path(__file__).resolve().parent
        preferred = root / "Newsheadlines" / "artifacts" / "news_b_augmented_ensemble.joblib"
        if preferred.exists():
            return preferred
        return root / "Newsheadlines" / "artifacts" / "news_b_char_lr.joblib"

    @staticmethod
    def _default_checkpoint_path() -> Path:
        return Path(__file__).resolve().parent / "model.pt"

    def _load_pipeline(self, weights_path: str | None) -> None:
        candidates: list[Path] = []
        if weights_path and weights_path != "__no_weights__.pth":
            candidates.append(Path(weights_path))
        candidates.append(self._default_checkpoint_path())

        for candidate in candidates:
            if candidate.exists() and candidate.suffix.lower() in {".pt", ".pth"}:
                checkpoint = torch.load(candidate, map_location="cpu")
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                if isinstance(checkpoint, dict) and "artifact_bytes" in checkpoint:
                    cleaned = {str(k).removeprefix("module.").removeprefix("model."): v for k, v in checkpoint.items()}
                    self.load_state_dict(cleaned, strict=False)
                    return

        if weights_path and weights_path != "__no_weights__.pth":
            candidate = Path(weights_path)
            if candidate.exists() and candidate.suffix.lower() in {".joblib", ".pkl"}:
                payload = joblib.load(candidate)
                self._set_pipeline_payload(payload)
                return

        artifact = self._default_artifact_path()
        if artifact.exists():
            payload = joblib.load(artifact)
            self._set_pipeline_payload(payload)
            return
        raise FileNotFoundError("No KK model artifact found. Include model.pt with model.py and preprocess.py.")

    def _set_pipeline_payload(self, payload: Any) -> None:
        self.decision_threshold = None
        self.decision_threshold_kind = None
        if isinstance(payload, dict) and "pipeline" in payload:
            self.pipeline = payload["pipeline"]
            meta = payload.get("meta") or {}
            if isinstance(meta, dict) and meta.get("decision_threshold") is not None:
                self.decision_threshold = float(meta["decision_threshold"])
                self.decision_threshold_kind = str(meta.get("decision_threshold_kind") or "proba")
            return
        self.pipeline = payload

    def _pipeline_from_buffers(self):
        n = int(self.artifact_len.detach().cpu().item())
        raw = bytes(self.artifact_bytes.detach().cpu().tolist()[:n])
        payload = joblib.load(io.BytesIO(raw))
        self._set_pipeline_payload(payload)
        return self.pipeline

    @staticmethod
    def _coerce_item_to_text(item: Any) -> str:
        if isinstance(item, dict):
            for key in ["headline", "title", "text", "content"]:
                if key in item and item[key]:
                    return normalize_text(str(item[key]))
            if "url" in item and item["url"]:
                return _MISSING_HEADLINE_TEXT
            return normalize_text(str(item))

        text = str(item)
        if text.startswith("http://") or text.startswith("https://"):
            return _MISSING_HEADLINE_TEXT
        return normalize_text(text)

    def predict(self, batch: Iterable[Any]) -> List[int]:
        if self.pipeline is None:
            raise RuntimeError("Model pipeline was not loaded.")
        texts = [self._coerce_item_to_text(x) for x in batch]
        if (
            self.decision_threshold is not None
            and self.decision_threshold_kind != "decision"
            and hasattr(self.pipeline, "predict_proba")
        ):
            classes = list(getattr(self.pipeline, "classes_", []))
            if 1 in classes:
                proba = self.pipeline.predict_proba(texts)
                one_idx = classes.index(1)
                return [int(p >= self.decision_threshold) for p in proba[:, one_idx]]
        if (
            self.decision_threshold is not None
            and self.decision_threshold_kind == "decision"
            and hasattr(self.pipeline, "decision_function")
        ):
            classes = list(getattr(self.pipeline, "classes_", []))
            if len(classes) == 2 and 1 in classes:
                scores = self.pipeline.decision_function(texts)
                if getattr(scores, "ndim", 1) == 1:
                    if classes[1] != 1:
                        scores = -scores
                else:
                    scores = scores[:, classes.index(1)]
                return [int(float(score) >= self.decision_threshold) for score in scores]
        preds = self.pipeline.predict(texts)
        return [int(p) for p in preds]


def get_model() -> Model:
    return Model()
