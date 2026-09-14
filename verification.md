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

## Environment

The checked environment used NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, joblib 1.5.3, and PyTorch 2.10.0, matching the requirement files.

The full ensemble search and historical shortcut experiments were not rerun. The report and experiment CSV preserve those original results.
