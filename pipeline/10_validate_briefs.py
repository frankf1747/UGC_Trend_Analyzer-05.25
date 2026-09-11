"""Stage 10 — Check every number in every brief against the evidence.

A generated narrative is only as trustworthy as its weakest sentence, and the
failure mode that matters is not an invented theme — those are obvious — but a
number that drifted. A prevalence quoted as 25% when the packet says 21%, a
drag of 0.16 written as 0.19: individually small, collectively fatal to trust,
and invisible to a reader who does not have the table open beside them.

So each brief is parsed for numeric claims, and every claim must match a value
the corresponding evidence packet actually contains, within rounding tolerance.
Anything unmatched is reported and must be corrected or removed before the
dashboard renders it.

This does not verify that a brief's *reasoning* is sound — no automated check
can. It verifies that the brief is arithmetically anchored to its evidence.
"""
import json, re
from config import EVIDENCE, BRIEFS, ARTIFACTS

REPORT = ARTIFACTS / "brief_validation.json"
TOL = 0.011                       # rounding slack on percentages and stars

def allowed_values(packet: dict, corpus: dict) -> set[float]:
    """Every number a brief about this category is permitted to cite."""
    vals: set[float] = set()
    s = packet["scope"]
    vals |= {float(s["reviews"]), float(s["sentences"]), float(s["products"]),
             float(s["mean_star"]), float(s["pct_negative_sentences"])}
    for t in packet["themes"]:
        vals |= {float(t["sentences"]), float(t["share_of_category_pct"]),
                 float(t["pct_negative"]), float(t["reviews_with_negative_mention"]),
                 round(100 * float(t["prevalence_of_complaint"]))}
        if "expected_drag_stars" in t:
            vals |= {abs(round(float(t["expected_drag_stars"]), 2)),
                     abs(float(t["coef_stars"]))}
    for t in packet.get("tailwinds", []):
        vals.add(abs(float(t["coef_stars"])))
    # corpus-level facts any brief may reference
    vals |= {float(v) for v in corpus["negative_share_by_star"].values()}
    vals |= {float(corpus["reviews"]), float(corpus["sentences"])}
    # documented pipeline constants a brief may legitimately name
    from config import MIN_SENTENCES_PER_CATEGORY
    vals |= {15.0, 40.0, float(MIN_SENTENCES_PER_CATEGORY)}
    # sums of the fix list are a legitimate derived claim
    drags = [abs(float(t["expected_drag_stars"])) for t in packet["fix_list"]]
    for i in range(2, len(drags) + 1):
        vals.add(round(sum(drags[:i]), 2))
        vals.add(round(sum(drags[1:i]), 2))
    return vals

NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(★|%|)")

def main() -> None:
    ev = json.loads(EVIDENCE.read_text())
    briefs = json.loads(BRIEFS.read_text())
    report, failures = {}, 0

    for cat, brief in briefs.items():
        packet = ev["categories"][cat]
        allowed = allowed_values(packet, ev["corpus"])
        unmatched = []
        for field, text in brief.items():
            for raw, _unit in NUM.findall(text):
                v = float(raw)
                if not any(abs(v - a) <= TOL for a in allowed):
                    unmatched.append({"field": field, "value": v})
        report[cat] = {"claims_checked": sum(len(NUM.findall(t)) for t in brief.values()),
                       "unmatched": unmatched}
        failures += len(unmatched)

    total = sum(r["claims_checked"] for r in report.values())
    REPORT.write_text(json.dumps({"total_claims": total, "unmatched": failures,
                                  "by_category": report}, indent=1))
    print(f"stage 10 · {total} numeric claims checked across {len(briefs)} briefs · "
          f"{failures} unmatched")
    for cat, r in report.items():
        if r["unmatched"]:
            print(f"  {cat}: {[u['value'] for u in r['unmatched']]}")
    if failures:
        raise SystemExit("briefs contain claims not supported by the evidence packets")

if __name__ == "__main__":
    main()
