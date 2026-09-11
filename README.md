# UGC Trend Analyzer

**Reads 4,173 customer reviews and says what to fix, what to feature, and what words to sell it in.**

July 2025 · Amazon Fine Food Reviews · Python, PyTorch, scikit-learn, statsmodels, Streamlit

---

## The finding that set the method

**17.5% of the sentences inside five-star reviews are complaints.** So are 26.5%
of those in four-star reviews.

The first version of this project, like most review-mining pipelines, took each
review's star rating and applied it to every sentence in that review. Under that
rule, "the packaging arrived crushed" inside a four-star review counts as praise.
Scoring each sentence independently shows that **21.9% of sentences carried the
wrong label** — and that the errors are not random. They are concentrated exactly
where a brand team would most want to look: in the complaints of otherwise
satisfied customers.

Satisfied people are not silent about problems. They bury them in good reviews,
which is precisely where a rating-derived method cannot see them.

## What it produces

For each of 14 product categories, two things:

**A ranked fix list.** Themes ordered by estimated drag on the star rating, not
by how often they are discussed. Volume rankings send teams after the loudest
complaint; this ranks by *how often it goes wrong here × how much it costs when
it does*, with 95% confidence intervals.

**The language to lead with.** The vocabulary that actually separates five-star
from one-star reviews in that category, by weighted log-odds with an informative
Dirichlet prior — not TF-IDF, which confuses frequency with distinctiveness.

Each category gets a short written brief where every number traces to a specific
figure in the evidence packet, checked automatically before it renders.

## Some of what it found

- **Complaining that you cannot find a product is associated with a rating 0.22★
  *higher*** (p = 0.03). Scarcity complaints come from devotees. In tea, where
  they are most common, they are a distribution signal — not a defect.
- **Protein snacks are the lowest-rated category (3.88★) because of logistics,
  not recipe.** Condition-and-expectation complaints reach 14% of reviews here,
  more than double any other category; 84.6% of those mentions are negative, and
  the distinguishing complaint words are "description", "date", "bag".
- **"Natural" is among the strongest markers of a *low*-rated drink review.** When
  a natural claim is what a reviewer reaches for while giving one star, the claim
  has been read as a promise about taste and missed.
- **The strongest complaint marker in baby & pet is "china".** Country of origin
  does real work in how owners judge a product, independent of the product.
- **Dairy is the only category where taste is not the leading fixable problem** —
  ingredients are. Its taste mentions are just 21% negative, against 28.8% for
  its ingredients — the reverse of every other category.

## How much to trust it

A pipeline that assigns labels with a model owes the reader an error rate.
On a stratified audit of 102 sentences judged against the written theme
definitions:

| | Accuracy | 95% CI (Wilson) |
|---|---|---|
| Theme assignment | **88.2%** | 80.6 – 93.1% |
| Polarity | **95.1%** | 89.0 – 97.9% |

That took three attempts, and the route matters more than the destination:

| Method | Theme accuracy |
|---|---|
| Clusters used directly as labels | 56.2% |
| Zero-shot entailment against theme definitions | 82.4% |
| — with an over-aggressive pooling rule removed | **88.2%** |

Each gain came from diagnosing a specific failure, not from tuning. The full
account — including the methods that were tried and rejected — is in
[`docs/METHOD.md`](docs/METHOD.md). Two are worth naming here:

**HDBSCAN, the obvious modern choice, failed.** On 9,954 short sentences it
returned 2 clusters and left 90% as noise. Keeping it would have meant throwing
away nine tenths of the evidence to preserve a fashionable method.

**Hypothesis wording mattered more than the model.** Feeding the model each
theme's full written definition left 62.5% of the corpus unassigned and lost
"Price & value" entirely — in a corpus of food reviews. Replacing the
definitions with short label phrases ("the price or value for money") dropped
that to 17.7% with no change of model, threshold, or data.

## What it cannot tell you

**Nothing causal.** Theme polarity is extracted from the same text that produced
the star rating, so the association is mechanical by construction: a reviewer who
complains about taste also rates low. These rankings are a triage order, not a
treatment effect. Nothing here supports "fix this and ratings will rise".

**Nothing about brands or products.** 3,438 products across 4,173 reviews is
about 1.2 reviews each. Product comparison is not supportable and is not
attempted.

**Nothing about today.** The data ends in 2012.

**Nothing about a representative sample.** The 4,173 reviews were drawn from the
568k-review source by a process the original project did not record. Every figure
describes this sample, and the method rather than the market.

Four of the 14 categories have too few low-rated reviews to support a vocabulary
comparison. The dashboard says so on those pages instead of showing a phrase list
built from thirteen reviews.

## The report

A three-page summary — the approach as a flow, and where the method transfers
to other industries — is at
[`docs/UGC_Trend_Analyzer_Report.pdf`](docs/UGC_Trend_Analyzer_Report.pdf).
Source for it is in `docs/report_src/`; it renders with headless Chrome.

## Running it

The pipeline runs once; `artifacts/` is committed. The dashboard needs no API
key, no model download, and no GPU:

```bash
pip install -r requirements.txt
make app
```

To recompute everything from the sentence corpus (~1 hour, mostly stage 05):

```bash
make pipeline
```

## Layout

```
pipeline/     01 prepare → 02 embed → 03 polarity → 04 discover themes
              05 assign → 06 audit labels → 07 drivers → 08 language
              09 evidence packets → 10 validate every claim in every brief
artifacts/    committed outputs — the dashboard reads only these
app/          Streamlit, narrative-first: claim on top, evidence beneath
docs/         METHOD.md — full method, and the parts that are weak
notebooks/    the July 2025 original, preserved for comparison
```

## Pipeline

| Stage | What it does |
|---|---|
| 01 Prepare | Cleans the corpus; discards embeddings the original saved via `str(ndarray)`, which truncated all 9,954 |
| 02 Embed | `BAAI/bge-small-en-v1.5`, local, MPS |
| 03 Polarity | Sentence-level RoBERTa, continuous score, replacing star inheritance |
| 04 Discover | Spherical k-means, k by silhouette → cluster exemplars |
| 05 Assign | Zero-shot NLI against a frozen, version-controlled taxonomy |
| 06 Audit | Stratified sample, judged, Wilson intervals |
| 07 Drivers | Review-level OLS, category and year fixed effects, HC1 errors |
| 08 Language | Weighted log-odds, informative Dirichlet prior (Monroe et al. 2008) |
| 09 Evidence | One packet per category — everything a brief may cite |
| 10 Validate | Parses every number in every brief; fails if one is unsupported |

Stage 10 currently checks **103 numeric claims across 14 briefs, 0 unmatched**.
It has caught real errors, including a figure summed by hand rather than taken
from the evidence.

It is not a guarantee of accuracy, and the honest version of that is in
`docs/METHOD.md`: the validator checks that a number exists in the evidence, not
that a *comparison* built on it is true. Two false superlatives passed it and
were caught only by checking against the artifacts by hand. Both underlying
numbers were right; both claims built on them were wrong.

---

*Built July 2025; rebuilt with a corrected method and published later. The
original notebook and dashboard are preserved under `notebooks/` — including the
API key that sat in them in plaintext for a year, redacted here and since
rotated, because that is a thing that happens in ordinary project code.*
