# Portfolio verification

Checked on September 14, 2026 in an isolated Python environment on macOS. These checks establish that the provided baseline and inference workflows run; they do not reproduce the private course leaderboard evaluation.

## Baseline reruns

Both scripts ran on the included `headlines.csv` with 3,802 rows, a stratified 3,041/761 training/validation split, and seed 42. Full-data retraining was disabled with `--no-final-retrain-on-full`.

| Baseline | Validation accuracy | Macro F1 |
|---|---:|---:|
| Word TF-IDF + logistic regression | 0.79369251 | 0.79231396 |
| Character TF-IDF + logistic regression | 0.83048620 | 0.82920150 |

The character result agrees with the reported 83.05%. The historical word result (80.03%) used a different, 3,803-row cleaned dataset; the current 79.37% rerun uses the stricter included 3,802-row dataset. These are distinct dataset variants, not two runs on an identical split.

## Checkpoint and inference

- The download helper successfully retrieved the checkpoint from the pinned upstream commit and verified its size (39,396,693 bytes) and SHA-256 against `model-manifest.json`.
- The inference helper successfully loaded the original checkpoint and returned predictions for two synthetic English headlines. This is a software smoke check, not an accuracy assessment.
- Python syntax, local Markdown links, and the preserved source-file hashes were checked.

## Fresh public-checkout verification

A new copy was cloned anonymously from the published repository at commit `96947465605b8a4464765c413c468c0c1090a0d9`. A fresh Python 3.11.14 virtual environment was created and the repository's `requirements.txt` installed successfully. Both README baseline commands completed, wrote their models to `artifacts/`, and reproduced the values above. The default `python download_model.py` command downloaded and verified the checkpoint, and the README's `python predict.py` example returned a prediction successfully.

## Dataset integrity checks

The included CSV contains 3,802 records: 2,000 Fox News (`0`) and 1,802 NBC News (`1`). Checks found:

- No missing headline, URL, source, or label fields.
- No invalid labels or inconsistencies between labels, source names, and URL domains.
- No duplicate URLs or duplicate headlines after case and whitespace normalization.
- No normalized headline assigned conflicting labels.
- No empty, symbol-only, or URL-only headline text.
- The dataset and other preserved source artifacts matched the hashes in `model-manifest.json`.

These checks cover structure and consistency. They do not constitute a manual review of every headline or establish that near-duplicate stories or other statistical biases are absent. The stored headline text is sufficient for the documented baseline runs; re-scraping the original news websites is unnecessary.

## Environment

The checked environment used NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, joblib 1.5.3, and PyTorch 2.10.0, matching the requirement files.

The full ensemble search and historical shortcut experiments were not rerun. The report and experiment CSV preserve those original results.
