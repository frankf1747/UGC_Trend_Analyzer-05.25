"""Stage 09 — Assemble one evidence packet per category.

This is the interface between analysis and narrative. Everything a brief is
allowed to claim must appear here first, with its sample size attached. Nothing
downstream reads the raw corpus, so no brief can assert something the evidence
does not contain.

The ranking idea worth explaining: the driver model in stage 07 is fitted
globally, because per-category models on ~300 reviews each would be too noisy to
trust. But a global effect size is not a per-category priority. A theme that
costs a lot per mention barely matters in a category where nobody mentions it.

So the fix list ranks by *expected drag*:

    expected_drag = P(negative mention | category) x |coefficient|

— how often this goes wrong here, times how much it costs when it does. The
confidence interval carries through from the coefficient, so a theme resting on
a weakly-estimated effect shows a wide band rather than a false precision.
"""
import json
import numpy as np, pandas as pd
from config import (THEMES, DRIVERS, LANGUAGE, EVIDENCE, VALIDATION,
                    MIN_SENTENCES_PER_CATEGORY)
from frames import mention_frame

N_QUOTES = 3
MIN_MENTIONS_FOR_RANKING = 15

def main() -> None:
    sent = pd.read_parquet(THEMES)
    mentions = mention_frame()
    drivers = pd.read_parquet(DRIVERS)
    lang = pd.read_parquet(LANGUAGE)
    validation = json.loads(VALIDATION.read_text())

    neg_effects = (drivers[drivers.direction == "negative mention"]
                   .set_index("theme")[["coef_stars", "ci_low", "ci_high", "p_value", "significant"]])

    analysed = mentions
    packets = {}

    for cat, grp in analysed.groupby("analysis_category"):
        n_sent, n_rev = len(grp), grp.review_idx.nunique()
        if n_sent < MIN_SENTENCES_PER_CATEGORY:
            continue

        themes = []
        for theme, tg in grp.groupby("theme"):
            rev_with = tg[tg.polarity_label == "negative"].review_idx.nunique()
            prevalence = rev_with / n_rev
            row = {"theme": theme,
                   "sentences": len(tg),
                   "share_of_category_pct": round(100 * len(tg) / n_sent, 1),
                   "pct_negative": round(100 * (tg.polarity_label == "negative").mean(), 1),
                   "mean_polarity": round(float(tg.polarity.mean()), 3),
                   "reviews_with_negative_mention": int(rev_with),
                   "prevalence_of_complaint": round(prevalence, 3)}
            if theme in neg_effects.index and rev_with >= MIN_MENTIONS_FOR_RANKING:
                e = neg_effects.loc[theme]
                row |= {"coef_stars": float(e.coef_stars),
                        "expected_drag_stars": round(prevalence * float(e.coef_stars), 4),
                        "drag_ci": [round(prevalence * float(e.ci_high), 4),
                                    round(prevalence * float(e.ci_low), 4)],
                        "effect_significant": bool(e.significant)}
            themes.append(row)

        # Only themes whose complaints are associated with a *lower* rating belong on
        # a fix list. Availability & sourcing has a significant positive coefficient —
        # complaining that you cannot find a product is something fans do — so it is
        # surfaced separately as a tailwind rather than as a defect to repair.
        scored = [t for t in themes if "expected_drag_stars" in t and t.get("effect_significant")]
        fixable = sorted([t for t in scored if t["expected_drag_stars"] < 0],
                         key=lambda t: t["expected_drag_stars"])
        tailwinds = sorted([t for t in scored if t["expected_drag_stars"] > 0],
                           key=lambda t: -t["expected_drag_stars"])

        # A sentence can legitimately carry several themes, but quoting it under each
        # of them reads as a bug and wastes the reader's attention. Each sentence
        # illustrates at most one theme, claimed by the higher-ranked fix.
        quotes, used = {}, set()
        for theme in [t["theme"] for t in fixable[:4]]:
            tg = grp[(grp.theme == theme) & (grp.polarity_label == "negative")
                     & (~grp.sent_id.isin(used))]
            picked = tg.nsmallest(N_QUOTES, "polarity")
            used.update(picked.sent_id.tolist())
            quotes[theme] = [{"sent_id": int(r.sent_id), "stars": int(r.Score),
                              "polarity": round(float(r.polarity), 2),
                              "text": r.sentence} for r in picked.itertuples()]

        cat_lang = lang[lang.category == cat]
        packets[cat] = {
            "scope": {"sentences": n_sent, "reviews": n_rev,
                      "products": int(grp.ProductId.nunique()),
                      "mean_star": round(float(grp.Score.mean()), 2),
                      "pct_negative_sentences": round(100 * (grp.polarity_label == "negative").mean(), 1),
                      "years": f"{int(grp.year.min())}–{int(grp.year.max())}"},
            "themes": sorted(themes, key=lambda t: -t["sentences"]),
            "fix_list": fixable[:5],
            "tailwinds": tailwinds[:3],
            "quotes": quotes,
            "praise_language": cat_lang[cat_lang.side == "praise"].head(10)[["phrase", "z"]].to_dict("records"),
            "complaint_language": cat_lang[cat_lang.side == "complaint"].head(10)[["phrase", "z"]].to_dict("records"),
        }

    corpus_level = {
        "sentences": int(len(sent)),
        "sentences_analysed": int(analysed.sent_id.nunique()),
        "theme_mentions": int(len(analysed)),
        "pct_unassigned": round(100 * (sent.theme == "Unassigned").mean(), 1),
        "reviews": int(sent.review_idx.nunique()),
        "products": int(sent.ProductId.nunique()),
        "categories_reported": len(packets),
        "label_validation": validation,
        "negative_share_by_star": {int(s): round(100 * float(v), 1) for s, v in
                                   sent.assign(n=sent.polarity_label.eq("negative"))
                                       .groupby("Score")["n"].mean().items()},
    }

    EVIDENCE.write_text(json.dumps({"corpus": corpus_level, "categories": packets}, indent=1))
    print(f"stage 09 · {len(packets)} category packets -> {EVIDENCE.name}")
    for cat, p in list(packets.items())[:4]:
        top = p["fix_list"][0]["theme"] if p["fix_list"] else "—"
        print(f"  {cat:18} {p['scope']['reviews']:4d} reviews · top fix: {top}")

if __name__ == "__main__":
    main()
