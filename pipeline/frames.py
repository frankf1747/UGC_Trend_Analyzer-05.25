"""Shared frame construction for the analysis stages.

Analysis runs on *mentions*, not on the single best-matching theme. A sentence
like "great flavour but the box arrived crushed" genuinely concerns two aspects,
and forcing it to one discards half of what the reviewer said. Taking the
argmax alone put 52% of all sentences under 'Taste & flavour', which flattens
every other theme against it.

`mention_frame()` is the long table every downstream stage reads: one row per
(sentence, theme) pair that cleared the entailment threshold, carrying the
sentence's polarity and its review metadata.
"""
import pandas as pd
from config import THEMES, ARTIFACTS

MENTIONS = ARTIFACTS / "theme_mentions.parquet"

def mention_frame() -> pd.DataFrame:
    sent = pd.read_parquet(THEMES)
    mentions = pd.read_parquet(MENTIONS)
    cols = ["sent_id", "review_idx", "sentence", "analysis_category", "Score",
            "ProductId", "year", "polarity", "polarity_label", "theme_actionable"]
    frame = mentions.merge(sent[cols].drop(columns=["theme_actionable"]), on="sent_id")
    import json
    from config import TAXONOMY
    tax = json.loads(TAXONOMY.read_text())["themes"]
    frame["theme_actionable"] = frame["theme"].map(lambda t: tax[t]["actionable"])
    return frame

def review_frame() -> pd.DataFrame:
    """One row per review, with its rating and metadata."""
    return (pd.read_parquet(THEMES)
              .groupby("review_idx")
              .agg(Score=("Score", "first"), category=("analysis_category", "first"),
                   year=("year", "first")))
