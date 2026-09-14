# Sources and preservation

- **Original public repository:** [Anthoneeee/ember-text-notes](https://github.com/Anthoneeee/ember-text-notes)
- **Upstream reference:** `1d3d4b5d8d4c5f200792944f21bb9feb50b6170d`
- **Public dataset:** [CIS 5190 Project B News Headlines](https://huggingface.co/datasets/Anthoneeee/cis5190-project-b-news-headlines)
- **Original team report:** `cis 5190 project report.pdf`, preserved here as [project-report.pdf](project-report.pdf)

## File mapping

`train_news_b_v1.py` is presented as `train_word_baseline.py`; `train_char_lr_final.py` is presented as `train_char_baseline.py`. Their contents are unchanged. `model.py`, `preprocess.py`, and `news_b_utils.py` are also unchanged.

`headlines.csv` preserves `deliverables/dataset/scraped_headlines_clean_headline_only.csv`. `shortcut-experiments.csv` preserves `deliverables/report/exploratory_results_step7.csv`. `shortcut-analysis.png` preserves `deliverables/figures/exploratory_cleaning_shortcut_step7.png`. File hashes are recorded in `model-manifest.json`.

The downloader pins the upstream commit and validates the checkpoint's SHA-256. The checkpoint's recorded course score is historical and has not been independently reproduced against the private course leaderboard.

## Reuse status

The original repository does not include an explicit license. No new license is applied to the original code, report, or datasets by this portfolio edition. Original authorship is retained; public availability should not be interpreted as an unrestricted reuse grant. News headlines and linked articles retain the rights of their respective publishers.
