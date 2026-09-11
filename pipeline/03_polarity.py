"""Stage 03 — Sentence-level polarity from a model, not from the star rating.

This replaces the central flaw of the 2025 pipeline, which took a review's star
rating and stamped it onto every sentence in that review. Under that scheme,
"the packaging arrived crushed" inside a 4-star review was filed as *positive
evidence*, and every downstream share-of-voice number inherited the error.

Here each sentence is scored independently with a RoBERTa classifier fine-tuned
on English review text. The output is continuous — P(positive) - P(negative) in
[-1, 1] — rather than a hard label, because sentence polarity genuinely is a
matter of degree and because the driver model in stage 07 can use the magnitude.

Sentences whose score falls inside a neutral band are marked neutral: a binary
classifier will otherwise force a polarity onto purely factual statements
("i ordered these in march"), which would be noise dressed as signal.
"""
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import CORPUS, POLARITY, POLARITY_MODEL

BATCH = 32
NEUTRAL_BAND = 0.40     # |score| below this => neutral

def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

def main() -> None:
    df = pd.read_parquet(CORPUS)
    dev = device()
    tok = AutoTokenizer.from_pretrained(POLARITY_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(POLARITY_MODEL).to(dev).eval()

    scores = []
    sents = df["sentence"].tolist()
    for i in range(0, len(sents), BATCH):
        batch = tok(sents[i:i + BATCH], padding=True, truncation=True,
                    max_length=256, return_tensors="pt").to(dev)
        with torch.no_grad():
            probs = torch.softmax(model(**batch).logits, dim=1)
        scores.append((probs[:, 1] - probs[:, 0]).cpu().numpy())   # 0=NEGATIVE, 1=POSITIVE
        if i % (BATCH * 40) == 0:
            print(f"  {i}/{len(sents)}", flush=True)

    df["polarity"] = np.concatenate(scores).astype(np.float32)
    df["polarity_label"] = np.select(
        [df.polarity >= NEUTRAL_BAND, df.polarity <= -NEUTRAL_BAND],
        ["positive", "negative"], default="neutral")

    # How wrong was the 2025 star-derived labelling? This is a finding, not a diagnostic.
    judged = df[df.polarity_label != "neutral"]
    agree = (judged.polarity_label == judged.star_sentiment).mean()
    flipped = judged[judged.polarity_label != judged.star_sentiment]
    by_star = (df.assign(neg=df.polarity_label == "negative")
                 .groupby("Score")["neg"].mean())

    df[["sent_id", "polarity", "polarity_label"]].to_parquet(POLARITY, index=False)
    print(f"stage 03 · scored {len(df)} sentences on {dev}")
    print(f"           label mix: {df.polarity_label.value_counts(normalize=True).round(3).to_dict()}")
    print(f"           agreement with 2025 star-derived sentiment: {agree:.1%}")
    print(f"           {len(flipped):,} sentences ({len(flipped)/len(judged):.1%} of non-neutral) were mislabelled")
    print(f"           share of sentences that are negative, by star rating:")
    for star, frac in by_star.items():
        print(f"             {star}★ {frac:6.1%}")

if __name__ == "__main__":
    main()
