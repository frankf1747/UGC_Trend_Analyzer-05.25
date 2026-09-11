# The 2025 original

Preserved for provenance, not for reuse. These are the files from the July 2025
build of this project, so the changes described in `docs/METHOD.md` can be
checked against what they replaced.

- `2025_original_pipeline.ipynb` — the original end-to-end notebook: keyword +
  LLM category labelling, TF-IDF keyphrases, HDBSCAN theme discovery with LLM
  cluster naming.
- `2025_original_app.py` — the original Streamlit dashboard.

Two things to know before reading them.

**An API key was committed in plaintext.** It appeared in the notebook source
and in the dashboard, and sat there for a year. Both copies here are redacted,
and the key has been rotated. It is left visible as a redaction marker rather
than silently removed, because the useful lesson is that this happens in
ordinary project code and survives every copy of the file.

**Sentiment was the star rating.** Each review's score was applied to every
sentence in it. Measuring this directly with a sentence-level model shows 21.9%
of sentences carried the wrong label, and that 17.5% of sentences inside
five-star reviews are complaints that the original method filed as praise.
