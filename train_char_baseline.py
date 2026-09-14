from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from news_b_utils import prepare_dataset_from_csv


def build_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_df=0.98,
                    max_features=30000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
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
    parser = argparse.ArgumentParser(description="Train final headline-only Char LR model.")
    parser.add_argument(
        "--input-csv",
        default="deliverables/dataset/scraped_headlines_clean_headline_only.csv",
        help="Headline-only cleaned dataset.",
    )
    parser.add_argument(
        "--output-model",
        default="Newsheadlines/artifacts/news_b_tfidf_lr.joblib",
        help="Output artifact path loaded by model.py.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-final-retrain-on-full",
        action="store_true",
        help="Keep validation-split model instead of retraining on full data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_dataset_from_csv(
        args.input_csv,
        allow_url_fallback=False,
        require_text_column=True,
        remove_duplicate_urls=True,
        remove_duplicate_headlines=True,
        min_headline_chars=8,
        drop_symbol_only_headlines=True,
    )
    x = prepared.texts
    y = np.asarray(prepared.labels, dtype=np.int64)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    val_pipeline = build_pipeline(args.random_state)
    val_pipeline.fit(x_train, y_train)
    val_pred = val_pipeline.predict(x_val)
    val_accuracy = float(accuracy_score(y_val, val_pred))
    val_macro_f1 = float(f1_score(y_val, val_pred, average="macro"))

    print(f"num_samples: {len(x)}")
    print(f"train_samples: {len(x_train)}")
    print(f"val_samples: {len(x_val)}")
    print(f"val_accuracy: {val_accuracy:.6f}")
    print(f"val_macro_f1: {val_macro_f1:.6f}")
    print("classification_report:")
    print(classification_report(y_val, val_pred, digits=4))

    final_retrain_on_full = not args.no_final_retrain_on_full
    if final_retrain_on_full:
        final_pipeline = build_pipeline(args.random_state)
        final_pipeline.fit(x, y)
        print("final_train_mode: full_dataset_retrain_after_validation")
    else:
        final_pipeline = val_pipeline
        print("final_train_mode: train_split_only_no_full_retrain")

    payload = {
        "pipeline": final_pipeline,
        "meta": {
            "model": "char_wb_tfidf_3_5+logistic_regression",
            "label_map": {"0": "FoxNews", "1": "NBC"},
            "num_samples": len(x),
            "train_samples": len(x_train),
            "val_samples": len(x_val),
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "final_retrain_on_full": final_retrain_on_full,
            "random_state": args.random_state,
            "test_size": args.test_size,
            "input_csv": args.input_csv,
            "tfidf": {
                "analyzer": "char_wb",
                "ngram_range": [3, 5],
                "min_df": 2,
                "max_df": 0.98,
                "max_features": 30000,
                "sublinear_tf": True,
                "strip_accents": "unicode",
                "lowercase": True,
            },
            "logistic_regression": {
                "max_iter": 2000,
                "solver": "liblinear",
                "C": 2.0,
                "class_weight": "balanced",
                "random_state": args.random_state,
            },
        },
    }

    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)
    print(f"saved_model: {output_path}")
    print("saved_meta_json:")
    print(json.dumps(payload["meta"], indent=2))


if __name__ == "__main__":
    main()
