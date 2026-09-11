"""Stage 08 — Which words actually distinguish praise from complaint.

Method: weighted log-odds ratio with an informative Dirichlet prior
(Monroe, Colaresi & Quinn 2008). TF-IDF, which the 2025 pipeline used, ranks
terms by frequency weighted by rarity — it cannot tell you whether a term is
*characteristic* of one group versus another, and it is wildly unstable for rare
terms, which is why TF-IDF keyphrase lists are so often dominated by noise that
happens to appear twice.

Weighted log-odds compares each term's usage between two corpora, shrinks the
estimate toward a background prior built from the whole corpus, and divides by
its own standard error. The output is a z-score: how confidently is this term
more characteristic of five-star reviews than one- and two-star reviews, given
how often we saw it at all. Rare terms shrink toward zero instead of exploding.
"""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from config import THEMES, LANGUAGE, MIN_SENTENCES_PER_CATEGORY

TOP_N = 25
PRIOR_STRENGTH = 500

def weighted_log_odds(counts_a: np.ndarray, counts_b: np.ndarray,
                      prior: np.ndarray) -> np.ndarray:
    """Monroe et al. (2008) eq. 22 — log-odds ratio, Dirichlet prior, z-scored."""
    n_a, n_b, n_p = counts_a.sum(), counts_b.sum(), prior.sum()
    odds_a = (counts_a + prior) / (n_a + n_p - counts_a - prior)
    odds_b = (counts_b + prior) / (n_b + n_p - counts_b - prior)
    delta = np.log(odds_a) - np.log(odds_b)
    var = 1.0 / (counts_a + prior) + 1.0 / (counts_b + prior)
    return delta / np.sqrt(var)

def main() -> None:
    sent = pd.read_parquet(THEMES)
    reviews = (sent.sort_values("sentence_idx")
                   .groupby("review_idx")
                   .agg(text=("sentence", " ".join), Score=("Score", "first"),
                        category=("analysis_category", "first")))

    rows = []
    for cat, grp in reviews.groupby("category"):
        hi, lo = grp[grp.Score >= 4], grp[grp.Score <= 2]
        if len(hi) < 25 or len(lo) < 25:
            print(f"  skipping {cat}: {len(hi)} praise / {len(lo)} complaint reviews — too few")
            continue
        vec = CountVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5,
                              stop_words="english", token_pattern=r"[a-z][a-z']+")
        X = vec.fit_transform(grp.text.str.lower())
        vocab = np.array(vec.get_feature_names_out())
        mask_hi = (grp.Score >= 4).values
        c_hi = np.asarray(X[mask_hi].sum(axis=0)).ravel().astype(float)
        c_lo = np.asarray(X[~mask_hi & (grp.Score <= 2).values].sum(axis=0)).ravel().astype(float)
        background = np.asarray(X.sum(axis=0)).ravel().astype(float)
        prior = background / background.sum() * PRIOR_STRENGTH

        z = weighted_log_odds(c_hi, c_lo, prior)
        order = np.argsort(z)
        for idx in order[::-1][:TOP_N]:
            rows.append({"category": cat, "side": "praise", "phrase": vocab[idx],
                         "z": round(float(z[idx]), 2), "n_praise": int(c_hi[idx]),
                         "n_complaint": int(c_lo[idx])})
        for idx in order[:TOP_N]:
            rows.append({"category": cat, "side": "complaint", "phrase": vocab[idx],
                         "z": round(float(z[idx]), 2), "n_praise": int(c_hi[idx]),
                         "n_complaint": int(c_lo[idx])})

    out = pd.DataFrame(rows)
    out.to_parquet(LANGUAGE, index=False)
    print(f"stage 08 · {out.category.nunique()} categories x {TOP_N} phrases per side")
    for cat in out.category.unique()[:3]:
        p = out[(out.category == cat) & (out.side == "praise")].head(6).phrase.tolist()
        c = out[(out.category == cat) & (out.side == "complaint")].head(6).phrase.tolist()
        print(f"  {cat:18} praise: {', '.join(p)}")
        print(f"  {'':18} complaint: {', '.join(c)}")

if __name__ == "__main__":
    main()
