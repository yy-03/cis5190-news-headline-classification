# Results and evaluation context

The tables below contain recorded team-project results, taken from the [original report](project-report.pdf) and [experiment table](shortcut-experiments.csv). Current portfolio baseline reruns are documented separately in [verification.md](verification.md).

## Submitted model and baselines

| Setting | Accuracy | Evaluation context |
|---|---:|---|
| Course-provided baseline | 66.49% | Historical baseline from the course handout; not rerun here on the same split |
| Word TF-IDF + logistic regression | 80.03% | Stratified internal split of the 3,803-row cleaned base dataset |
| Character TF-IDF + logistic regression | 83.05% | Stratified internal split of the 3,802-row headline-only dataset |
| Final heterogeneous ensemble | 87.42% | Best recorded course Hugging Face leaderboard result |

The final model was selected through repeated leaderboard feedback. The leaderboard result and internal-split results answer different evaluation questions and should not be treated as a single controlled comparison. Full-corpus sanity-check scores are intentionally excluded from the performance table because the model was trained on those data.

## Shortcut analysis

All rows below use the team's recorded 3,042/761 train/test split and corresponding input transformations.

| Training representation | Evaluation representation | Accuracy |
|---|---|---:|
| Headline | Headline | 80.03% |
| URL | URL | 96.71% |
| URL | Headline | 54.40% |
| Headline + URL | Headline + URL | 95.66% |
| Headline + URL | Headline | 65.31% |

The decrease after changing the available input demonstrates interface sensitivity and shortcut reliance. The final pipeline therefore uses URLs for retrieving headlines, rather than passing URL strings to the classifier.

## Cleaning stress test

| Condition | Less strict cleaning | Strict cleaning |
|---|---:|---:|
| Recorded base data | 80.03% | 80.03% |
| Injected conflicting duplicates | 75.95% | 80.03% |

The stress test deliberately adds conflicting examples. Its improvement does not establish that every cleaning operation improves accuracy on naturally occurring data.

## Reproduction scope

The included scripts allow retraining of the word and character baselines and inference with the frozen final model. They do not reconstruct every historical ensemble-search or shortcut-analysis run. The experiment CSV and original report preserve those recorded results. The private course evaluation data are not included.
