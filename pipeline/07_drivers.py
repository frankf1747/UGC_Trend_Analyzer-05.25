"""Stage 07 — Which themes move the star rating, and by how much.

Share of voice is not importance. A theme can dominate the conversation and
barely move the rating; a rarer theme can be the one that loses the customer.
The 2025 version ranked themes by share of voice and called the result a
recommendation, which is how teams end up fixing the loudest complaint instead
of the costliest one.

Model: review-level OLS of star rating on the presence of each actionable theme
in each polarity, with category and year fixed effects and HC1 robust standard
errors. The unit is the review (n≈4,100) because the rating is a review-level
outcome; treating each sentence as an observation would inflate significance by
counting one opinion several times.

IMPORTANT — this is association, not causation. Theme polarity is extracted from
the same text that produced the rating, so the relationship is mechanical by
construction: a reviewer who complains about taste also gives a low star rating,
and nothing here establishes that fixing taste would raise ratings. The
coefficients rank where dissatisfaction concentrates. They are a triage order,
not a treatment effect, and the README says so in those words.
"""
import numpy as np, pandas as pd
import statsmodels.api as sm
from config import DRIVERS
from frames import mention_frame, review_frame

MIN_REVIEWS_PER_FEATURE = 40

def main() -> None:
    df = mention_frame()
    df = df[df.theme_actionable]

    # Review-level design matrix: does this review mention theme T negatively / positively?
    flags = (df.assign(neg=df.polarity_label.eq("negative"),
                       pos=df.polarity_label.eq("positive"))
               .groupby(["review_idx", "theme"])[["neg", "pos"]].max()
               .unstack(fill_value=False))
    flags.columns = [f"{pol}::{theme}" for pol, theme in flags.columns]

    data = review_frame().join(flags, how="left").fillna(False)

    keep = [c for c in flags.columns if data[c].sum() >= MIN_REVIEWS_PER_FEATURE]
    dropped = sorted(set(flags.columns) - set(keep))
    X = data[keep].astype(float)
    X = pd.concat([X,
                   pd.get_dummies(data.category, prefix="cat", drop_first=True).astype(float),
                   pd.get_dummies(data.year, prefix="yr", drop_first=True).astype(float)], axis=1)
    X = sm.add_constant(X)
    model = sm.OLS(data.Score.astype(float), X).fit(cov_type="HC1")

    rows = []
    for feat in keep:
        pol, theme = feat.split("::")
        ci = model.conf_int().loc[feat]
        rows.append({"theme": theme,
                     "direction": "negative mention" if pol == "neg" else "positive mention",
                     "coef_stars": round(float(model.params[feat]), 3),
                     "ci_low": round(float(ci[0]), 3), "ci_high": round(float(ci[1]), 3),
                     "p_value": float(model.pvalues[feat]),
                     "n_reviews": int(data[feat].sum()),
                     "significant": bool(model.pvalues[feat] < 0.05)})
    out = pd.DataFrame(rows).sort_values("coef_stars")
    out.to_parquet(DRIVERS, index=False)

    print(f"stage 07 · n={int(model.nobs)} reviews · adj R²={model.rsquared_adj:.3f} · "
          f"{len(keep)} theme features ({len(dropped)} dropped for n<{MIN_REVIEWS_PER_FEATURE})")
    show = out.copy()
    show["p_value"] = show.p_value.map(lambda p: f"{p:.1e}")
    print(show.to_string(index=False))

if __name__ == "__main__":
    main()
