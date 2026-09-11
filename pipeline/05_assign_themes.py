"""Stage 05 — Assign sentences to themes by zero-shot entailment.

Why not just use the clusters from stage 04?

Because clustering and assignment are different jobs, and using one model for
both is what broke the first version of this stage. An audit of 96 sentences put
cluster-derived theme labels at 56% accuracy (CI95 46–66%). Reading the failures
showed why: k-means must place *every* sentence somewhere, so pure chatter
("i am 61 years old") lands in a theme, and clusters organise partly by surface
style — "i love X" sentences group together regardless of what X is. Filtering by
distance to centroid barely helped: discarding 40% of the corpus lifted accuracy
only to 67%, which is a poor trade.

So clustering keeps the job it is good at — *discovering* which themes exist,
which is how `taxonomy.json` was written — and assignment moves to a natural
language inference model scored against the written theme definitions. This
handles the two things clustering could not: a sentence can carry more than one
aspect, and a sentence can carry none.

Sentences where no theme clears the threshold are marked `Unassigned` and are
excluded from every downstream statistic rather than being quietly absorbed.
"""
import json
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import CORPUS, POLARITY, TAXONOMY, THEMES, ARTIFACTS

MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
THRESHOLD = 0.50          # entailment probability required to count as a mention
PAIR_BATCH = 256          # (sentence, hypothesis) pairs scored per forward pass
MENTIONS = ARTIFACTS / "theme_mentions.parquet"

def main() -> None:
    df = pd.read_parquet(CORPUS).merge(pd.read_parquet(POLARITY), on="sent_id")
    tax = json.loads(TAXONOMY.read_text())["themes"]
    names = list(tax)
    # Short `label` phrases, not the long `definition`. Feeding the definitions in
    # as hypotheses left 62.5% of the corpus unassigned and lost 'Price & value'
    # entirely — compound clauses entail poorly. Short labels: 17.7% unassigned.
    hypotheses = [f"This sentence is about {tax[n]['label']}." for n in names]

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()
    entail_idx = model.config.label2id.get("entailment", 0)

    sents = df["sentence"].tolist()
    n_s, n_h = len(sents), len(hypotheses)
    scores = np.zeros((n_s, n_h), dtype=np.float32)

    # Flatten to (sentence, hypothesis) pairs and batch them directly. The HF
    # zero-shot pipeline scores one sentence at a time, which leaves the GPU
    # mostly idle; batching pairs is ~20x faster on the same hardware.
    cache = ARTIFACTS / "theme_scores.npy"
    if cache.exists() and np.load(cache).shape == (n_s, n_h):
        print("  reusing cached entailment scores (delete theme_scores.npy to recompute)")
        scores = np.load(cache)
        return finish(df, tax, names, scores)

    pairs = [(i, j) for i in range(n_s) for j in range(n_h)]
    for start in range(0, len(pairs), PAIR_BATCH):
        chunk = pairs[start:start + PAIR_BATCH]
        enc = tok([sents[i] for i, _ in chunk], [hypotheses[j] for _, j in chunk],
                  padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
        with torch.no_grad():
            logits = model(**enc).logits
        # Binary entailment vs not — softmax over the two-class head of this model.
        probs = torch.softmax(logits, dim=1)[:, entail_idx].float().cpu().numpy()
        for (i, j), p in zip(chunk, probs):
            scores[i, j] = p
        if start % (PAIR_BATCH * 100) == 0:
            print(f"  {start}/{len(pairs)} pairs", flush=True)

    np.save(ARTIFACTS / "theme_scores.npy", scores)
    return finish(df, tax, names, scores)


def finish(df, tax, names, scores) -> None:
    best = scores.argmax(axis=1)
    df["primary_score"] = scores.max(axis=1)
    df["theme"] = [names[b] for b in best]
    df.loc[df.primary_score < THRESHOLD, "theme"] = "Unassigned"
    df["theme_actionable"] = df["theme"].map(lambda t: tax.get(t, {}).get("actionable", False))

    # Multi-label mentions: one row per (sentence, theme) above threshold.
    si, ti = np.where(scores >= THRESHOLD)
    mentions = pd.DataFrame({"sent_id": df.sent_id.values[si],
                             "theme": [names[t] for t in ti],
                             "score": scores[si, ti]})
    mentions.to_parquet(MENTIONS, index=False)

    # No pooling of small themes. An earlier version merged any theme with fewer
    # than 30 primary assignments into 'Other', which swallowed 'Strength &
    # balance' — a coherent theme about coffee being too strong or too bitter —
    # and turned every one of its sentences into a labelling error. Small themes
    # are kept; the sample-size guards live downstream, where stage 07 requires
    # 40 reviews per modelled feature and stage 09 requires 15 before ranking.

    df.to_parquet(THEMES, index=False)
    unassigned = (df.theme == "Unassigned").mean()
    print(f"stage 05 · {len(df)} sentences · {unassigned:.1%} unassigned (no theme cleared {THRESHOLD})")
    print(f"           {len(mentions)} theme mentions · {len(mentions)/len(df):.2f} per sentence")
    summary = (df[df.theme != "Unassigned"]
               .groupby("theme").agg(n=("sent_id", "size"),
                                     mean_polarity=("polarity", "mean"),
                                     pct_negative=("polarity_label", lambda s: (s == "negative").mean())))
    summary["pct_negative"] = (summary.pct_negative * 100).round(1)
    summary["mean_polarity"] = summary.mean_polarity.round(3)
    print(summary.sort_values("n", ascending=False).to_string())

if __name__ == "__main__":
    main()
