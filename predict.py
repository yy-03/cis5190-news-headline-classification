"""Run headline-only inference with the submitted course ensemble."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("headlines", nargs="+", help="One or more headline strings")
    parser.add_argument("--weights", type=Path, default=Path(__file__).resolve().parent / "model.pt")
    args = parser.parse_args()
    if any(not text.strip() or text.strip().lower().startswith(("http://", "https://")) for text in args.headlines):
        parser.error("Provide non-empty headline text, not article URLs.")
    if not args.weights.is_file():
        parser.error("Checkpoint not found. Run python download_model.py first.")
    from model import Model
    model = Model(weights_path=str(args.weights))
    labels = model.predict(args.headlines)
    names = {0: "Fox News", 1: "NBC News"}
    for headline, label in zip(args.headlines, labels):
        print(json.dumps({"headline": headline, "predicted_label": label,
                          "predicted_source": names[label]}, ensure_ascii=False))

if __name__ == "__main__":
    main()
