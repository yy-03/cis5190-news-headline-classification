from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from news_b_utils import prepare_dataset_from_csv

_TEXT_COL_CANDIDATES = ("headline", "title", "text", "content", "news_title")


def _build_pipeline(max_features: int, ngram_max: int, random_state: int) -> Pipeline:
    """Build a consistent TF-IDF + LogisticRegression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, ngram_max),
                    min_df=2,
                    max_df=0.98,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    C=2.0,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project B training script (TF-IDF + LR)")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="Newsheadlines/scraped_headlines_clean.csv",
        help="Input CSV path (recommended: cleaned CSV with real scraped headlines)",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="Newsheadlines/artifacts/news_b_tfidf_lr.joblib",
        help="Output model artifact path",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--max-features", type=int, default=30000, help="Maximum TF-IDF feature count")
    parser.add_argument("--ngram-max", type=int, default=2, help="Upper bound for TF-IDF n-grams")
    parser.add_argument(
        "--allow-url-pseudo-text",
        action="store_true",
        help="Allow URL-derived pseudo text when headline columns are missing (off by default)",
    )
    parser.add_argument(
        "--no-final-retrain-on-full",
        action="store_true",
        help="Disable final full-dataset retraining after validation",
    )
    parser.add_argument(
        "--remove-duplicate-urls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove duplicate URLs during dataset preparation (default: enabled)",
    )
    parser.add_argument(
        "--remove-duplicate-headlines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove duplicate headlines during dataset preparation (default: enabled)",
    )
    parser.add_argument(
        "--min-headline-chars",
        type=int,
        default=8,
        help="Drop headlines shorter than this length after normalization (default: 8)",
    )
    parser.add_argument(
        "--drop-symbol-only-headlines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop headlines that do not contain letters or digits (default: enabled)",
    )
    return parser.parse_args()


def _assert_real_headline_input(csv_path: str, allow_url_pseudo_text: bool) -> None:
    """
    By default, require real headline/title/text content for training inputs.
    """
    if allow_url_pseudo_text:
        return

    df = pd.read_csv(csv_path)
    lower_map = {c.lower(): c for c in df.columns}
    text_col = None
    for cand in _TEXT_COL_CANDIDATES:
        if cand in lower_map:
            text_col = lower_map[cand]
            break

    if text_col is None:
        raise ValueError(
            "Input CSV does not include headline/title/text columns. "
            "Use Newsheadlines/scraped_headlines_clean.csv or explicitly pass "
            "--allow-url-pseudo-text."
        )

    non_empty = df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")
    if int(non_empty.sum()) == 0:
        raise ValueError(
            f"Text column {text_col!r} is empty. Please verify the input contains real headlines."
        )


def main() -> None:
    args = parse_args()
    _assert_real_headline_input(args.input_csv, args.allow_url_pseudo_text)

    # Step 1: load and prepare text samples + labels
    prepared = prepare_dataset_from_csv(
        args.input_csv,
        allow_url_fallback=args.allow_url_pseudo_text,
        require_text_column=not args.allow_url_pseudo_text,
        remove_duplicate_urls=args.remove_duplicate_urls,
        remove_duplicate_headlines=args.remove_duplicate_headlines,
        min_headline_chars=args.min_headline_chars,
        drop_symbol_only_headlines=args.drop_symbol_only_headlines,
    )
    X = prepared.texts
    y = np.asarray(prepared.labels, dtype=np.int64)

    # Step 2: stratified train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # Step 3: fit on training split for validation metrics
    val_pipeline = _build_pipeline(
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        random_state=args.random_state,
    )
    val_pipeline.fit(X_train, y_train)

    # Step 4: report core validation metrics
    y_pred = val_pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    report = classification_report(y_val, y_pred, digits=4)

    print(f"num_samples: {len(X)}")
    print(f"train_samples: {len(X_train)}")
    print(f"val_samples: {len(X_val)}")
    print(f"val_accuracy: {acc:.6f}")
    print(f"val_macro_f1: {macro_f1:.6f}")
    print(
        "cleaning_config: "
        f"remove_duplicate_urls={args.remove_duplicate_urls}, "
        f"remove_duplicate_headlines={args.remove_duplicate_headlines}, "
        f"min_headline_chars={args.min_headline_chars}, "
        f"drop_symbol_only_headlines={args.drop_symbol_only_headlines}"
    )
    print("classification_report:")
    print(report)

    # Step 5: by default, retrain on full dataset for final export
    final_retrain_on_full = not args.no_final_retrain_on_full
    if final_retrain_on_full:
        final_pipeline = _build_pipeline(
            max_features=args.max_features,
            ngram_max=args.ngram_max,
            random_state=args.random_state,
        )
        final_pipeline.fit(X, y)
        print("final_train_mode: full_dataset_retrain_after_validation")
    else:
        final_pipeline = val_pipeline
        print("final_train_mode: train_split_only_no_full_retrain")

    # Step 6: export model and metadata for inference loading
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": final_pipeline,
        "meta": {
            "model": "tfidf+logistic_regression",
            "label_map": {"0": "FoxNews", "1": "NBC"},
            "num_samples": len(X),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "val_accuracy": float(acc),
            "val_macro_f1": float(macro_f1),
            "final_retrain_on_full": final_retrain_on_full,
            "random_state": args.random_state,
            "test_size": args.test_size,
            "max_features": args.max_features,
            "ngram_max": args.ngram_max,
            "remove_duplicate_urls": args.remove_duplicate_urls,
            "remove_duplicate_headlines": args.remove_duplicate_headlines,
            "min_headline_chars": args.min_headline_chars,
            "drop_symbol_only_headlines": args.drop_symbol_only_headlines,
        },
    }
    joblib.dump(payload, output_path)
    print(f"saved_model: {output_path}")
    print("saved_meta_json:")
    print(json.dumps(payload["meta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
