"""UGC Trend Analyzer — narrative-first review intelligence.

The dashboard reads committed artifacts only. It never calls a model, never
needs an API key, and produces the same numbers on every machine. Everything it
shows was computed by `pipeline/` and frozen in `artifacts/`.

Layout follows the argument rather than the data model: the brief states a
claim, and the chart that justifies it sits directly underneath. Charts are not
collected into a gallery for the reader to interpret unaided.
"""
from __future__ import annotations
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

st.set_page_config(page_title="UGC Trend Analyzer", layout="wide",
                   initial_sidebar_state="expanded")

INK, MUTED, NEG, POS = "#1b1b1b", "#6b6b6b", "#b4412c", "#2f6b52"

st.markdown("""
<style>
  .block-container {max-width: 1120px; padding-top: 2.2rem;}
  h1, h2, h3 {letter-spacing: -0.015em;}
  .lede {font-size: 1.12rem; line-height: 1.62; color: #2b2b2b; max-width: 68ch;}
  .claim {font-size: 1.02rem; line-height: 1.6; max-width: 70ch; margin: 0.4rem 0 0.9rem 0;}
  .quote {border-left: 3px solid #d8d2c8; padding: 0.35rem 0 0.35rem 0.9rem;
          margin: 0.45rem 0; color: #3a3a3a; font-size: 0.94rem; line-height: 1.55;}
  .quote span {color: #8a8a8a; font-size: 0.82rem;}
  .caveat {background: #faf8f4; border: 1px solid #e8e2d8; border-radius: 6px;
           padding: 0.85rem 1rem; font-size: 0.9rem; color: #4a4a4a; line-height: 1.55;}
  .metric-row {display: flex; gap: 2.6rem; margin: 0.6rem 0 1.4rem 0;}
  .metric b {display: block; font-size: 1.6rem; font-weight: 600; color: #1b1b1b;}
  .metric span {font-size: 0.8rem; color: #6b6b6b; text-transform: uppercase;
                letter-spacing: 0.06em;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    ev = json.loads((ART / "evidence_packets.json").read_text())
    briefs = json.loads((ART / "briefs.json").read_text()) if (ART / "briefs.json").exists() else {}
    import sys
    sys.path.insert(0, str(ROOT / "pipeline"))
    from frames import mention_frame
    return ev, briefs, mention_frame(), pd.read_parquet(ART / "drivers.parquet")


def metrics(pairs: list[tuple[str, str]]) -> None:
    html = "".join(f"<div class='metric'><b>{v}</b><span>{k}</span></div>" for k, v in pairs)
    st.markdown(f"<div class='metric-row'>{html}</div>", unsafe_allow_html=True)


def fix_chart(fix_list: list[dict]) -> alt.Chart:
    df = pd.DataFrame(fix_list)
    df["drag"] = -df.expected_drag_stars           # plot as positive magnitude
    # drag_ci arrives as [upper, lower] in signed star terms (both negative);
    # negating flips the order, so name the bounds after negation, not before.
    ci = pd.DataFrame(df.drag_ci.tolist(), columns=["signed_hi", "signed_lo"])
    df["lo"] = -ci.signed_hi     # smaller magnitude of drag
    df["hi"] = -ci.signed_lo     # larger magnitude of drag
    base = alt.Chart(df).encode(
        y=alt.Y("theme:N", sort="-x", title=None,
                axis=alt.Axis(labelFontSize=12, labelColor=INK, labelLimit=220)))
    bars = base.mark_bar(size=17, color=NEG, opacity=0.85).encode(
        x=alt.X("drag:Q", title="Estimated drag on category star rating",
                axis=alt.Axis(format=".2f")),
        tooltip=[alt.Tooltip("theme", title="Theme"),
                 alt.Tooltip("drag:Q", title="Expected drag (stars)", format=".3f"),
                 alt.Tooltip("prevalence_of_complaint:Q", title="Complaint prevalence", format=".1%"),
                 alt.Tooltip("coef_stars:Q", title="Cost per complaint (stars)", format=".2f"),
                 alt.Tooltip("reviews_with_negative_mention:Q", title="Reviews complaining")])
    err = base.mark_rule(strokeWidth=1.4, color=INK, opacity=0.55).encode(x="lo:Q", x2="hi:Q")
    return (bars + err).properties(height=28 * len(df) + 40)


def coverage_chart(df: pd.DataFrame, cat: str) -> alt.Chart:
    d = (df[(df.analysis_category == cat)]
         .groupby("theme")
         .agg(sentences=("sent_id", "size"), pct_neg=("polarity_label", lambda s: (s == "negative").mean()))
         .reset_index())
    return (alt.Chart(d).mark_circle(opacity=0.85)
            .encode(x=alt.X("pct_neg:Q", title="Share of mentions that are negative",
                            axis=alt.Axis(format="%")),
                    y=alt.Y("sentences:Q", title="Mentions in category"),
                    size=alt.Size("sentences:Q", legend=None, scale=alt.Scale(range=[60, 600])),
                    color=alt.Color("pct_neg:Q", legend=None,
                                    scale=alt.Scale(scheme="redyellowgreen", reverse=True)),
                    tooltip=["theme", "sentences",
                             alt.Tooltip("pct_neg:Q", title="% negative", format=".1%")])
            .properties(height=330))


def language_table(rows: list[dict], side: str) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    d.columns = ["Phrase", "Distinctiveness (z)"]
    return d


ev, briefs, sent, drivers = load()
corpus, cats = ev["corpus"], ev["categories"]

st.sidebar.markdown("### UGC Trend Analyzer")
st.sidebar.caption("Amazon Fine Food reviews · July 2025")
page = st.sidebar.radio("View", ["Start here", "Category briefs", "Method & limits"],
                        label_visibility="collapsed")

# ───────────────────────────────────────────── Start here
if page == "Start here":
    st.title("What customers actually said")
    st.markdown(
        f"<p class='lede'>{corpus['reviews']:,} product reviews, broken into "
        f"{corpus['sentences']:,} sentences, each scored for what it is about and how "
        f"positive it is — then read for what a brand team should do about it.</p>",
        unsafe_allow_html=True)

    v = corpus["label_validation"]
    metrics([("Reviews", f"{corpus['reviews']:,}"),
             ("Sentences analysed", f"{corpus['sentences_analysed']:,}"),
             ("Categories", str(corpus["categories_reported"])),
             ("Theme label accuracy", f"{v['theme_accuracy']:.0%}"),
             ("Polarity accuracy", f"{v['polarity_accuracy']:.0%}")])

    st.subheader("The finding that changed the method")
    neg_by_star = pd.DataFrame({"stars": [int(k) for k in corpus["negative_share_by_star"]],
                                "pct": list(corpus["negative_share_by_star"].values())})
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(
            f"<p class='claim'>The 2025 version of this project took a review's star rating "
            f"and applied it to every sentence in that review. Scoring each sentence "
            f"independently shows why that fails: "
            f"<b>{corpus['negative_share_by_star']['5']:.0f}% of sentences inside five-star "
            f"reviews are complaints</b>, and {corpus['negative_share_by_star']['4']:.0f}% of "
            f"those in four-star reviews. Every one of them was previously filed as praise.</p>",
            unsafe_allow_html=True)
        st.markdown(
            "<p class='claim'>Satisfied customers are not silent about problems. They "
            "bury them in otherwise positive reviews — which is exactly where a "
            "rating-based method cannot see them.</p>", unsafe_allow_html=True)
    with right:
        st.altair_chart(
            alt.Chart(neg_by_star).mark_bar(size=34, color=NEG, opacity=0.85).encode(
                x=alt.X("stars:O", title="Star rating"),
                y=alt.Y("pct:Q", title="% of sentences that are complaints"),
                tooltip=["stars", alt.Tooltip("pct:Q", format=".1f")]
            ).properties(height=300), use_container_width=True)

    st.subheader("Where the damage concentrates, across all categories")
    d = drivers[(drivers.direction == "negative mention") & drivers.significant].nsmallest(8, "coef_stars")
    st.altair_chart(
        alt.Chart(d).mark_bar(size=18, color=NEG, opacity=0.85).encode(
            x=alt.X("coef_stars:Q", title="Association with star rating (stars)"),
            y=alt.Y("theme:N", sort="x", title=None,
                    axis=alt.Axis(labelLimit=220)),
            tooltip=["theme", alt.Tooltip("coef_stars:Q", format=".2f"),
                     alt.Tooltip("n_reviews:Q", title="Reviews")]
        ).properties(height=300), use_container_width=True)
    st.markdown(
        "<div class='caveat'><b>Read this as triage, not treatment.</b> Theme polarity is "
        "extracted from the same text that produced the star rating, so these associations "
        "are mechanical by construction. They tell you where dissatisfaction concentrates. "
        "They do not establish that fixing a theme would raise ratings.</div>",
        unsafe_allow_html=True)

# ───────────────────────────────────────────── Category briefs
elif page == "Category briefs":
    order = sorted(cats, key=lambda c: -cats[c]["scope"]["reviews"])
    cat = st.sidebar.selectbox("Category", order,
                               format_func=lambda c: f"{c}  ({cats[c]['scope']['reviews']} reviews)")
    p = cats[cat]
    brief = briefs.get(cat, {})

    st.title(cat.title())
    if brief.get("headline"):
        st.markdown(f"<p class='lede'>{brief['headline']}</p>", unsafe_allow_html=True)

    s = p["scope"]
    metrics([("Reviews", f"{s['reviews']:,}"), ("Sentences", f"{s['sentences']:,}"),
             ("Products", f"{s['products']:,}"), ("Mean rating", f"{s['mean_star']:.2f}★"),
             ("Complaint sentences", f"{s['pct_negative_sentences']:.0f}%")])

    if p["fix_list"]:
        st.subheader("What to fix first")
        if brief.get("fix_argument"):
            st.markdown(f"<p class='claim'>{brief['fix_argument']}</p>", unsafe_allow_html=True)
        st.altair_chart(fix_chart(p["fix_list"]), use_container_width=True)
        st.caption("Expected drag = how often customers complain about this here × how much "
                   "that complaint is associated with, per review. Bars show the 95% interval.")

        st.subheader("In their words")
        for theme, quotes in list(p["quotes"].items())[:3]:
            st.markdown(f"**{theme}**")
            for q in quotes:
                st.markdown(
                    f"<div class='quote'>“{q['text']}”<br><span>{q['stars']}★ review · "
                    f"polarity {q['polarity']:+.2f} · sentence #{q['sent_id']}</span></div>",
                    unsafe_allow_html=True)
    else:
        st.info("No theme in this category cleared both the sample-size floor and "
                "statistical significance. That is a finding: nothing here is reliably "
                "separable from noise at this sample size.")

    if p.get("tailwinds"):
        st.subheader("Not a problem")
        for t in p["tailwinds"]:
            st.markdown(
                f"<p class='claim'><b>{t['theme']}</b> — complaints here are associated with a "
                f"rating <b>{t['coef_stars']:+.2f}★</b>, not lower "
                f"({t['reviews_with_negative_mention']} reviews). A complaint that travels with "
                f"satisfaction is a signal about demand, not a defect to repair.</p>",
                unsafe_allow_html=True)

    st.subheader("What to lead with")
    if brief.get("language_argument"):
        st.markdown(f"<p class='claim'>{brief['language_argument']}</p>", unsafe_allow_html=True)
    if not p["praise_language"]:
        st.markdown(
            "<div class='caveat'>No distinguishing-language estimate is reported for this "
            "category. Weighted log-odds needs both groups to be large enough to compare; "
            "with fewer than 25 low-rated reviews here, any phrase list would be noise "
            "presented as insight.</div>", unsafe_allow_html=True)
        st.subheader("Theme landscape")
        st.altair_chart(coverage_chart(sent, cat), use_container_width=True)
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Language of praise**")
        st.dataframe(language_table(p["praise_language"], "praise"),
                     hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Language of complaint**")
        st.dataframe(language_table(p["complaint_language"], "complaint"),
                     hide_index=True, use_container_width=True)
    st.caption("Weighted log-odds with an informative Dirichlet prior (Monroe et al. 2008). "
               "Higher |z| = more characteristic of that group, adjusted for how often the "
               "phrase appears at all.")

    st.subheader("Theme landscape")
    st.altair_chart(coverage_chart(sent, cat), use_container_width=True)
    st.caption("Upper-left is loud and well-liked. Lower-right is quiet but sour — "
               "small, angry themes that volume rankings miss.")

# ───────────────────────────────────────────── Method
else:
    st.title("Method & limits")
    st.markdown((ROOT / "docs" / "METHOD.md").read_text())
