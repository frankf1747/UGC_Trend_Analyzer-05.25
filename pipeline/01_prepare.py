"""Stage 01 — Build a clean sentence corpus from the preserved 2025 extract.

Two things happen here that matter for everything downstream:

1. The `embedding` column of the source file is dropped. It was written with
   `str(numpy_array)`, which truncates to `[0.02 -0.03 ... 0.01]` — every one of
   the 9,954 rows is unrecoverable. Stage 02 recomputes them.

2. The `sentiment` column is *kept but renamed* to `star_sentiment`. In the 2025
   pipeline this was derived from the review's star rating and stamped onto every
   sentence in that review, so a complaint inside a 4-star review was labelled
   positive. Stage 03 replaces it with a model score; keeping the old column lets
   us quantify how wrong it was.
"""
import pandas as pd
from config import SOURCE_SENTENCES, CORPUS, RANDOM_SEED

def main() -> None:
    df = pd.read_csv(SOURCE_SENTENCES)
    n_raw = len(df)

    df = df.drop(columns=[c for c in ("embedding", "cluster_id", "cluster_conf",
                                      "high_theme", "sub_theme", "len") if c in df.columns])
    df = df.rename(columns={"sentiment": "star_sentiment"})

    df["sentence"] = df["sentence"].astype(str).str.strip()
    df = df[df["sentence"].str.split().str.len() >= 4]          # drop fragments
    df = df.drop_duplicates(subset=["review_idx", "sentence"])

    df["review_dt"] = pd.to_datetime(df["review_dt"], errors="coerce", utc=True)
    df["year"] = df["review_dt"].dt.year
    df["period"] = df["review_dt"].dt.to_period("Y").astype(str)
    df = df.dropna(subset=["review_dt", "Score", "analysis_category"])
    df["Score"] = df["Score"].astype(int)

    df = df.reset_index(drop=True)
    df["sent_id"] = df.index

    df.to_parquet(CORPUS, index=False)
    print(f"stage 01 · {n_raw} raw rows -> {len(df)} sentences")
    print(f"           {df.review_idx.nunique()} reviews · "
          f"{df.ProductId.nunique()} products · {df.analysis_category.nunique()} categories")
    print(f"           {df.review_dt.min().date()} -> {df.review_dt.max().date()}")
    print(f"           wrote {CORPUS.relative_to(CORPUS.parent.parent)}")

if __name__ == "__main__":
    main()
