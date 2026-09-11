# app.py
import os, json, math
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# ---------- Config ----------
COV_CSV  = "themes_coverage_summary.csv"     # from Step 4 (coverage table)
LBL_CSV  = "themes_sentences_labeled.csv"    # from Step 4 (one row per sentence with high_theme)
KP_CSV   = "keyphrases_by_sentiment.csv"     # from Step 4 (tidy keyphrases)
INSIGHTS_JSON = "insights_cache.json"        # local cache
OPENAI_MODEL = "gpt-4o-mini"

# ---------- Optional OpenAI client ----------
client = None
try:
    from openai import OpenAI
    client = OpenAI(api_key="<REDACTED-ROTATE-THIS-KEY>")
except Exception:
    client = None  # run fine without LLM

# ---------- Streamlit setup ----------
st.set_page_config(page_title="UGC Trend Analyzer", layout="wide")
st.title("UGC Trend Analyzer — Themes & Phrases")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_all():
    cov = _load_csv(COV_CSV).copy()
    lbl = _load_csv(LBL_CSV).copy()
    kp  = _load_csv(KP_CSV).copy()

    # Normalize columns (robust to prior scripts)
    if 'high_theme' not in cov.columns and 'theme' in cov.columns:
        cov = cov.rename(columns={'theme': 'high_theme'})
    if 'high_theme' not in lbl.columns and 'theme' in lbl.columns:
        lbl = lbl.rename(columns={'theme': 'high_theme'})

    # Ensure required columns exist
    for c in ['sentiment','analysis_category','high_theme']:
        if c not in cov.columns and not lbl.empty:
            # recompute coverage fresh from lbl
            cov = None
            break

    # (Re)compute accurate share_pct from LBL to avoid >100% issues
    if lbl.empty:
        cov = pd.DataFrame(columns=['sentiment','analysis_category','high_theme','sentences','total_sents','share_pct'])
    else:
        # drop obvious dupes if any
        lbl = lbl.drop_duplicates(subset=['sentiment','analysis_category','sentence','high_theme'], keep='first')
        # coverage per (sentiment, category, high_theme)
        cov2 = (lbl
            .groupby(['sentiment','analysis_category','high_theme'])
            .size()
            .reset_index(name='sentences'))
        # totals per (sentiment, category)
        tots = cov2.groupby(['sentiment','analysis_category'])['sentences']\
                   .sum().reset_index(name='total_sents')
        cov2 = cov2.merge(tots, on=['sentiment','analysis_category'], how='left')
        cov2['share_pct'] = (100 * cov2['sentences'] / cov2['total_sents']).round(1)
        cov = cov2

    # Parse year_month for trends if present
    if 'year_month' in lbl.columns:
        try:
            lbl['year_month'] = pd.to_datetime(lbl['year_month'])
        except Exception:
            pass

    # Keyphrases tidy expectations
    if not kp.empty:
        # expected columns: analysis_category, sentiment, rank, phrase, cat_docs
        # try to normalize a bit:
        kp_cols = {c.lower(): c for c in kp.columns}
        # lower-case to find keys, then rename to canonical
        canon = {}
        for want in ['analysis_category','sentiment','rank','phrase','cat_docs']:
            found = None
            for c in kp.columns:
                if c.lower() == want:
                    found = c
                    break
            if found:
                canon[found] = want
        kp = kp.rename(columns=canon)
        # Keep only those columns we need
        keep = [c for c in ['analysis_category','sentiment','rank','phrase','cat_docs'] if c in kp.columns]
        kp = kp[keep].copy()

    return cov, lbl, kp

@st.cache_data(show_spinner=False)
def get_filters_from_data(cov: pd.DataFrame):
    sentiments = ["positive","negative"]
    if 'sentiment' in cov.columns:
        sentiments = sorted(cov['sentiment'].dropna().unique().tolist())
    cats = []
    if 'analysis_category' in cov.columns:
        cats = sorted(cov['analysis_category'].dropna().unique().tolist())
    return sentiments, cats

def _multi_select_all(label, options, default_all=True, key=None):
    if not options:
        return []
    default = options if default_all else []
    chosen = st.sidebar.multiselect(label, options, default=default, key=key)
    if not chosen:
        st.sidebar.info(f"Nothing selected for {label}. Falling back to ALL.")
        chosen = options
    return chosen

@st.cache_data(show_spinner=False)
def _load_insights_cache() -> dict:
    if os.path.exists(INSIGHTS_JSON):
        try:
            with open(INSIGHTS_JSON, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

@st.cache_data(show_spinner=False)
def _save_insights_cache(data: dict):
    with open(INSIGHTS_JSON, "w") as f:
        json.dump(data, f, indent=2)
    return True

def _generate_insight_text(df_slice: pd.DataFrame, sent: str, cat: str) -> str:
    """
    Create a small data summary text to feed the LLM (or use directly when no LLM).
    """
    if df_slice.empty:
        return f"[{sent} | {cat}] No theme data available."

    top = (df_slice.sort_values('share_pct', ascending=False)
           .head(8)[['high_theme','share_pct','sentences']])
    bullet = "\n".join([f"- {r.high_theme}: {r.share_pct:.1f}% ({r.sentences} sentences)" for r in top.itertuples()])
    txt = f"""Category: {cat} | Sentiment: {sent}
Top themes (share within this sentiment & category):
{bullet}"""
    return txt

def _llm_summarize(df_slice: pd.DataFrame, sent: str, cat: str) -> str:
    """
    Summarize insights with the LLM (cached per (sent,cat)). If client is None, return a rule-based summary.
    """
    cache = _load_insights_cache()
    key = f"{sent}__{cat}"
    if key in cache:
        return cache[key]

    # Fallback summary when no client
    if client is None:
        text = _generate_insight_text(df_slice, sent, cat)
        baseline = "LLM unavailable — using rule-based notes.\n"
        # naive two bullets:
        recs = []
        if not df_slice.empty:
            top1 = df_slice.sort_values('share_pct', ascending=False).head(1)
            if not top1.empty:
                th = top1.iloc[0]['high_theme']
                recs.append(f"Double down on '{th}' — it’s the top driver.")
        if len(recs) < 2:
            recs.append("Improve clarity in product pages and packaging to reduce negative 'Other'.")
        out = baseline + text + "\n\nRecommendations:\n- " + "\n- ".join(recs)
        cache[key] = out
        _save_insights_cache(cache)
        return out

    # Build compact prompt for this one pair
    sample_text = _generate_insight_text(df_slice, sent, cat)
    prompt = f"""
You are a product insights analyst. Based only on the theme coverage below,
produce: (1) a 3–4 sentence insight summary; (2) 3 actionable recommendations.

{sample_text}

Output format:
Summary:
- ...
- ...
Recommendations:
- ...
- ...
    """.strip()

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content": prompt}],
            temperature=0.4
        )
        text = resp.choices[0].message.content
    except Exception as e:
        text = "LLM error — falling back.\n" + _generate_insight_text(df_slice, sent, cat)

    cache[key] = text
    _save_insights_cache(cache)
    return text

def _bar_share_chart(df: pd.DataFrame, title: str):
    if df.empty: return None
    base = alt.Chart(df).mark_bar().encode(
        x=alt.X('share_pct:Q', title='Share (%)', scale=alt.Scale(domain=[0,100])),
        y=alt.Y('high_theme:N', sort='-x', title='Theme'),
        color=alt.Color('high_theme:N', legend=None)
    ).properties(height=400, title=title)
    return base

def _heatmap(df: pd.DataFrame, title: str):
    if df.empty: return None
    # show top themes per category by share
    chart = (alt.Chart(df)
             .mark_rect()
             .encode(
                x=alt.X('analysis_category:N', title='Category', sort=alt.SortField('analysis_category')),
                y=alt.Y('high_theme:N', title='Theme'),
                color=alt.Color('share_pct:Q', title='Share (%)', scale=alt.Scale(domain=[0,100])),
                tooltip=['analysis_category','high_theme','share_pct','sentences']
             ).properties(height=420, title=title))
    return chart

def _trend_chart(lbl: pd.DataFrame, cat: str, sent: str):
    if lbl.empty or 'year_month' not in lbl.columns:
        return None
    df = lbl[(lbl['analysis_category']==cat) & (lbl['sentiment']==sent)].copy()
    if df.empty or df['year_month'].isna().all():
        return None
    # monthly theme share
    m = (df.groupby(['year_month','high_theme']).size()
         .reset_index(name='sentences'))
    tot = m.groupby(['year_month'])['sentences'].sum().reset_index(name='total')
    m = m.merge(tot, on='year_month')
    m['share_pct'] = (100 * m['sentences'] / m['total']).round(1)
    # Keep top 6 themes overall
    top6 = (m.groupby('high_theme')['sentences'].sum()
              .sort_values(ascending=False).head(6).index.tolist())
    m = m[m['high_theme'].isin(top6)]
    line = (alt.Chart(m)
            .mark_line(point=True)
            .encode(
                x=alt.X('year_month:T', title='Month'),
                y=alt.Y('share_pct:Q', title='Share (%)', scale=alt.Scale(domain=[0,100])),
                color=alt.Color('high_theme:N', title='Theme'),
                tooltip=['year_month','high_theme','share_pct','sentences']
            ).properties(height=350, title=f"Theme share over time — {cat} | {sent}"))
    return line

# ---------- Load data ----------
cov, lbl, kp = load_all()

# ---------- Sidebar filters ----------
sentiments, categories = get_filters_from_data(cov if not cov.empty else lbl)
sentiment_opt = st.sidebar.selectbox("Sentiment", ["All"] + sentiments, index=0)
cats_sel = _multi_select_all("Categories", categories, default_all=True, key="cats")

# apply filters
cov_f = cov.copy()
lbl_f = lbl.copy()
kp_f  = kp.copy()

if sentiment_opt != "All":
    cov_f = cov_f[cov_f['sentiment'] == sentiment_opt]
    lbl_f = lbl_f[lbl_f['sentiment'] == sentiment_opt]
    if not kp_f.empty and 'sentiment' in kp_f.columns:
        kp_f = kp_f[kp_f['sentiment'] == sentiment_opt]

if cats_sel:
    cov_f = cov_f[cov_f['analysis_category'].isin(cats_sel)]
    lbl_f = lbl_f[lbl_f['analysis_category'].isin(cats_sel)]
    if not kp_f.empty and 'analysis_category' in kp_f.columns:
        kp_f = kp_f[kp_f['analysis_category'].isin(cats_sel)]

# ---------- High-level metrics ----------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Sentences", f"{len(lbl_f):,}")
with c2:
    cats_cnt = lbl_f['analysis_category'].nunique() if not lbl_f.empty else 0
    st.metric("Categories", cats_cnt)
with c3:
    themes_cnt = lbl_f['high_theme'].nunique() if not lbl_f.empty else 0
    st.metric("Themes", themes_cnt)
with c4:
    avg_len = lbl_f['sentence'].str.split().map(len).mean() if 'sentence' in lbl_f.columns and not lbl_f.empty else 0
    st.metric("Avg sentence length", f"{avg_len:.1f} words")

# ---------- Tabs ----------
tab_overview, tab_category, tab_phrases, tab_examples, tab_insights = st.tabs(
    ["Overview", "Category Deep-Dive", "Keyphrases", "Examples", "Insights"]
)

# ====== Overview ======
with tab_overview:
    st.subheader("Theme share across categories")
    # top themes by share (averaged within category selection)
    if cov_f.empty:
        st.info("No coverage data available.")
    else:
        # Show top 10 themes across selection
        top = (cov_f.groupby('high_theme')['share_pct'].mean()
                .sort_values(ascending=False).head(10).reset_index())
        ch = _bar_share_chart(top, title="Top themes by average share (%) within selected categories")
        if ch: st.altair_chart(ch, use_container_width=True)

        # Heatmap (category x theme)
        # keep top 8 themes per category for readability
        hm_df = (cov_f.sort_values(['analysis_category','share_pct'], ascending=[True,False])
                      .groupby('analysis_category').head(8))
        heat = _heatmap(hm_df, title="Theme share heatmap (top 8 themes per category)")
        if heat: st.altair_chart(heat, use_container_width=True)

# ====== Category Deep-Dive ======
with tab_category:
    st.subheader("Deep-Dive")
    if not categories:
        st.info("No categories available.")
    else:
        cat_pick = st.selectbox("Category", options=categories, index=0)
        sent_pick = sentiment_opt if sentiment_opt != "All" else st.selectbox("Sentiment", options=sentiments, index=0)

        cdf = cov[(cov['analysis_category']==cat_pick) & (cov['sentiment']==sent_pick)].copy()
        if cdf.empty:
            st.info("No coverage for this selection.")
        else:
            # Top themes bar
            bar_df = cdf.sort_values('share_pct', ascending=False).head(12)
            bar = _bar_share_chart(bar_df[['high_theme','share_pct']], title=f"Top themes — {cat_pick} | {sent_pick}")
            if bar: st.altair_chart(bar, use_container_width=True)

            # Trend (if time exists)
            trend = _trend_chart(lbl, cat_pick, sent_pick)
            if trend: st.altair_chart(trend, use_container_width=True)
            else: st.caption("No monthly timestamps available for trend.")

            # Lower-level themes (if present)
            if 'lower_theme' in lbl.columns:
                st.markdown("##### Lower-level themes (examples)")
                low = (lbl[(lbl['analysis_category']==cat_pick) & (lbl['sentiment']==sent_pick)]
                        .groupby(['high_theme','lower_theme']).size().reset_index(name='sentences')
                        .sort_values(['high_theme','sentences'], ascending=[True,False]))
                st.dataframe(low.head(50), use_container_width=True)

# ====== Keyphrases ======
with tab_phrases:
    st.subheader("Keyphrases by category & sentiment")
    if kp_f.empty:
        st.info("No keyphrase file found or empty.")
    else:
        kp_sent = sentiment_opt if sentiment_opt != "All" else st.selectbox("Sentiment (keyphrases)", options=sorted(kp['sentiment'].unique()), index=0, key="kp_sent")
        kp_cat = st.selectbox("Category (keyphrases)", options=sorted(kp[kp['sentiment']==kp_sent]['analysis_category'].unique()), key="kp_cat")
        topn = st.slider("Top N phrases to show", min_value=5, max_value=30, value=15, step=1)
        view = kp[(kp['sentiment']==kp_sent) & (kp['analysis_category']==kp_cat)].copy()
        if view.empty:
            st.info("No phrases for this selection.")
        else:
            view = view.sort_values('rank').head(topn)
            st.dataframe(view[['rank','phrase','cat_docs']], use_container_width=True)
            # small bar chart of ranks reversed (lower rank = more important)
            try:
                chart = alt.Chart(view).mark_bar().encode(
                    x=alt.X('rank:O', sort='ascending', title='Rank (1 = top)'),
                    y=alt.Y('cat_docs:Q', title='Docs in category (cohort)'),
                    tooltip=['phrase','rank','cat_docs']
                ).properties(height=300, title=f"Keyphrases — {kp_cat} | {kp_sent}")
                st.altair_chart(chart, use_container_width=True)
            except Exception:
                pass

# ====== Examples ======
with tab_examples:
    st.subheader("Example sentences")
    if lbl_f.empty:
        st.info("No labeled sentences available.")
    else:
        e_sent = sentiment_opt if sentiment_opt != "All" else st.selectbox("Sentiment (examples)", options=sentiments, index=0, key="ex_sent")
        e_cat = st.selectbox("Category (examples)", options=sorted(lbl[lbl['sentiment']==e_sent]['analysis_category'].unique()), key="ex_cat")
        e_themes = sorted(lbl[(lbl['sentiment']==e_sent) & (lbl['analysis_category']==e_cat)]['high_theme'].dropna().unique())
        e_theme = st.selectbox("Theme", options=e_themes, index=0)
        n_show = st.slider("How many examples?", 5, 100, 25, step=5)
        sample = (lbl[(lbl['sentiment']==e_sent) & (lbl['analysis_category']==e_cat) & (lbl['high_theme']==e_theme)]
                  [['analysis_category','sentiment','high_theme','sentence']]
                  .head(n_show))
        st.dataframe(sample, use_container_width=True)

# ====== Insights (LLM, cached to disk) ======
with tab_insights:
    st.subheader("LLM Insights (cached locally)")
    st.caption("Insights are cached to insights_cache.json. Regenerate only if your data changed.")

    if cov_f.empty:
        st.info("No coverage data to summarize.")
    else:
        # Decide which pairs to summarize
        pairs = (cov_f[['sentiment','analysis_category']]
                 .drop_duplicates()
                 .sort_values(['sentiment','analysis_category'])
                 .itertuples(index=False, name=None))

        regen = st.checkbox("Regenerate insights for shown pairs (uses OpenAI if key present)", value=False)
        st.write("")

        for sent, cat in pairs:
            df_slice = cov[(cov['sentiment']==sent) & (cov['analysis_category']==cat)].copy()
            if regen:
                # bypass cache by deleting key
                cache = _load_insights_cache()
                k = f"{sent}__{cat}"
                if k in cache:
                    del cache[k]
                    _save_insights_cache(cache)
            txt = _llm_summarize(df_slice, sent, cat)
            st.markdown(f"##### {cat} — {sent}")
            st.write(txt)
            st.divider()

# ---------- Footer ----------
st.caption("© 2025 UGC Trend Analyzer • Theme shares recomputed from sentence-level labels to ensure accuracy (0–100%).")
