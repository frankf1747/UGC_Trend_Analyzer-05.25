"""Stage 06 — Emit a stratified audit sample, and score the returned verdicts.

Any pipeline that assigns labels with a model owes the reader an error rate.
This stage draws a stratified random sample (`--emit`), which is judged
independently against the taxonomy definitions, and then scores the verdicts
(`--score`) into `validation.json` for the README to quote.

Judging is a separate, recorded step rather than an automatic one: a pipeline
that grades its own homework in the same pass is not a validation, it is a
formatting exercise.
"""
import json, sys
import pandas as pd
from config import THEMES, TAXONOMY, VALIDATION, ARTIFACTS, RANDOM_SEED

SAMPLE_PATH = ARTIFACTS / "validation_sample.json"
VERDICT_PATH = ARTIFACTS / "validation_verdicts.json"
PER_THEME = 6

def emit() -> None:
    df = pd.read_parquet(THEMES)
    sample = (df.groupby("theme", group_keys=False)
                .apply(lambda g: g.sample(min(PER_THEME, len(g)), random_state=RANDOM_SEED)))
    rows = [{"sent_id": int(r.sent_id), "theme": r.theme,
             "polarity_label": r.polarity_label, "polarity": round(float(r.polarity), 3),
             "sentence": r.sentence} for r in sample.itertuples()]
    SAMPLE_PATH.write_text(json.dumps(rows, indent=1))
    print(f"emitted {len(rows)} sentences for audit -> {SAMPLE_PATH.name}")

def score() -> None:
    verdicts = {int(k): v for k, v in json.loads(VERDICT_PATH.read_text()).items()}
    sample = {r["sent_id"]: r for r in json.loads(SAMPLE_PATH.read_text())}
    n = len(verdicts)
    theme_ok = sum(1 for k, v in verdicts.items() if v["theme_ok"])
    pol_ok = sum(1 for k, v in verdicts.items() if v["polarity_ok"])

    per_theme: dict[str, dict] = {}
    for k, v in verdicts.items():
        t = sample[k]["theme"]
        d = per_theme.setdefault(t, {"n": 0, "theme_ok": 0})
        d["n"] += 1
        d["theme_ok"] += int(v["theme_ok"])

    # Wilson 95% interval — the normal approximation is unreliable at n=96.
    def wilson(k: int, n: int) -> list[float]:
        if n == 0:
            return [0.0, 0.0]
        p, z = k / n, 1.96
        d = 1 + z**2 / n
        c = (p + z**2 / (2 * n)) / d
        h = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / d
        return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]

    out = {
        "n_audited": n,
        "theme_accuracy": round(theme_ok / n, 3),
        "theme_accuracy_ci95": wilson(theme_ok, n),
        "polarity_accuracy": round(pol_ok / n, 3),
        "polarity_accuracy_ci95": wilson(pol_ok, n),
        "per_theme": {t: {"n": d["n"], "accuracy": round(d["theme_ok"] / d["n"], 2)}
                      for t, d in sorted(per_theme.items())},
        "weakest_themes": sorted(((round(d["theme_ok"] / d["n"], 2), t) for t, d in per_theme.items()))[:3],
    }
    VALIDATION.write_text(json.dumps(out, indent=1))
    print(f"theme accuracy  {out['theme_accuracy']:.1%}  CI95 {out['theme_accuracy_ci95']}")
    print(f"polarity accuracy {out['polarity_accuracy']:.1%}  CI95 {out['polarity_accuracy_ci95']}")
    print(f"weakest: {out['weakest_themes']}")

if __name__ == "__main__":
    (emit if "--emit" in sys.argv else score)()
