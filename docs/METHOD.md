# Method

Every number the dashboard shows was produced by `pipeline/` and frozen into
`artifacts/`. This document says how, and — more usefully — where the method
is weak.

## Corpus

A 4,173-review extract of the Amazon Fine Food Reviews dataset, split into
9,954 sentences across 14 product categories, spanning November 2003 to
October 2012.

The extract was drawn during the original July 2025 project by a sampling
process that was not recorded. This matters and is not recoverable: the sample
may not be representative of the 568k-review source, so every figure here
describes *this sample*, not Amazon's food category. Ratings skew heavily
positive — 64% five-star — which is typical of voluntary review data and means
complaint themes are estimated from a smaller base than praise themes.

## Stage 01 — Corpus preparation

Drops sentences shorter than four words, de-duplicates within a review, and
discards the source file's `embedding` column. That column was unrecoverable:
it had been written with `str(numpy_array)`, so every one of the 9,954 vectors
was stored in NumPy's truncated display form — `[0.0258 -0.0311 ... 0.0170]`.

The original star-derived sentiment column is kept under the name
`star_sentiment`, so stage 03 can measure how wrong it was.

## Stage 02 — Embeddings

`BAAI/bge-small-en-v1.5`, CLS pooling, L2-normalised, on Apple MPS. 384
dimensions. Used for theme *discovery* only — not for assignment.

## Stage 03 — Polarity

`siebert/sentiment-roberta-large-english` scores each sentence independently.
Output is continuous: `P(positive) − P(negative)`, in [−1, 1].

This replaces the central flaw of the 2025 pipeline, which assigned each
review's star rating to every sentence in that review. Under that scheme a
complaint inside a four-star review counted as praise.

**Known weakness.** The model is binary and highly confident, so the neutral
band (|score| < 0.40) captures only 0.2% of sentences. Genuinely neutral
statements — "i ordered these in march" — are therefore forced into a polarity.
The audit below quantifies the resulting error rate rather than hiding it. A
three-class or ordinal model would handle this better and is the first thing
to change if this work continued.

## Stage 04 — Theme discovery

Spherical k-means over the sentence embeddings, k chosen by silhouette across
{24, 32, 40, 48, 56}. k=24 won.

**Density clustering was tried first and rejected.** HDBSCAN — the BERTopic
default and the obvious modern choice — returned 2 clusters and left 90% of
sentences as noise. Short review sentences do not form density peaks in this
embedding space. Keeping the fashionable method would have meant discarding 90%
of the evidence.

Silhouette scores were low in absolute terms (0.047 at the chosen k). Text
embeddings rarely produce well-separated clusters, so this number is weak
evidence of quality; the label audit below is the real check.

## Stage 05 — Theme assignment

Sentences are assigned to themes by **zero-shot entailment**
(`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`) scored against the written theme
definitions in `artifacts/taxonomy.json`, multi-label, threshold 0.50.
Sentences where nothing clears the threshold are marked `Unassigned` and
excluded from every downstream statistic.

**Hypothesis wording changed the result more than the model did.** The first
run fed each theme's full written definition to the NLI model as its hypothesis
— "This sentence is about price against grocery/competitor alternatives; whether
the quantity justifies the cost." That left **62.5% of the corpus unassigned**
and shrank "Price & value" to under 30 sentences, in a corpus of food reviews
where price is discussed constantly. Compound, multi-clause hypotheses entail
badly: the model is asked to accept the whole conjunction at once.

Replacing them with short label phrases — "the price or value for money" —
dropped unassigned sentences to **17.7%** at the identical threshold, with no
change of model, threshold, or data. The taxonomy therefore carries two strings
per theme: a short `label` used as the entailment hypothesis, and a longer
`definition` used for documentation and for human auditing.

**Why assignment was separated from discovery.** The first version of this
stage used the clusters directly as labels. An audit of 96 sentences put that
approach at **56% theme accuracy** (95% CI 46–66%). The failure mode was
visible on inspection: k-means must place every sentence somewhere, so pure
chatter ("i am 61 years old") landed inside a theme, and clusters organised
partly by phrasing — "i love X" sentences grouped together regardless of what
X was. Gating on distance to centroid barely helped: discarding 40% of the
corpus raised accuracy only to 67%.

So clustering keeps the job it does well — discovering which themes exist,
which is how the taxonomy was written — and assignment moved to a model that
compares a sentence against an explicit definition, can assign several themes
to one sentence, and can assign none.

## Taxonomy

Sixteen aspect themes, listed with definitions in `artifacts/taxonomy.json`.

Two design choices are worth stating. **Themes are aspects, not sentiments**:
"the flavour is wonderful" and "the flavour is vile" are one theme with
opposite polarity, because folding sentiment into the theme name makes it
impossible to ask whether an aspect is improving. **Two themes are marked
non-actionable** — "Overall satisfaction" and "Loyalty & repeat purchase" are
restatements of the rating rather than causes of it, and are excluded from the
fix-list ranking to avoid predicting the outcome with the outcome.

The taxonomy was written once, by an LLM reading cluster exemplars, then frozen
and version-controlled. It is not regenerated at runtime, so any change to a
theme shows up in a diff.

## Stage 06 — Label audit

A stratified random sample is drawn and judged against the taxonomy definitions.
Both theme assignment and polarity are graded. Accuracy is reported with a
Wilson 95% interval, which is appropriate at n≈96 where the normal
approximation is not.

Reported in the dashboard and in the README. A pipeline that assigns labels
with a model owes the reader an error rate.

## Stage 07 — Drivers

Review-level OLS: star rating on the presence of each actionable theme in each
polarity, with category and year fixed effects, HC1 robust standard errors.
Features appearing in fewer than 40 reviews are dropped.

The unit is the review, not the sentence, because the rating is a review-level
outcome; treating sentences as independent observations would count one opinion
several times and inflate significance.

**This is association, not causation, and the distinction is not pedantic
here.** Theme polarity is extracted from the same text that produced the star
rating, so the relationship is mechanical by construction: a reviewer who
complains about taste also gives a low rating. Nothing in this design supports
the claim that fixing an issue would raise ratings. The coefficients rank where
dissatisfaction concentrates — a triage order, not a treatment effect.

## Stage 08 — Distinguishing language

Weighted log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi &
Quinn 2008), comparing 4–5 star against 1–2 star reviews within each category.

TF-IDF, used by the 2025 version, ranks terms by frequency weighted by rarity.
It cannot say whether a term is *characteristic* of one group against another,
and it is unstable for rare terms — which is why TF-IDF keyphrase lists fill up
with noise that happened to appear twice. Weighted log-odds shrinks estimates
toward a corpus-wide prior and divides by their own standard error, so rare
terms move toward zero instead of exploding.

## Stage 09 — Evidence packets

One packet per category, containing every number a brief is permitted to cite,
with sample sizes attached. Nothing downstream reads the corpus.

The fix list ranks by **expected drag**:

```
expected_drag = P(negative mention | category) × coefficient
```

The driver model is fitted globally, because per-category models on ~300
reviews would be too noisy to trust. But a global effect size is not a local
priority — a theme that is costly per mention barely matters where nobody
mentions it. Multiplying the global cost by local prevalence gives a
category-specific ranking resting on a stably estimated effect. The confidence
interval carries through from the coefficient.

Themes are ranked only where at least 15 reviews complain about them and the
underlying coefficient is significant at p < 0.05.

## Stage 10 — Claim validation

Every brief is parsed for numeric claims, and each must match a value its
category's evidence packet actually contains, within rounding tolerance.
Anything unmatched fails the stage. It currently checks 103 claims across 14
briefs with none unmatched, and it has caught genuine errors — including a
figure the author summed by hand instead of taking from the evidence.

**What it does not catch, and this is the important part.** The validator checks
that each number *exists* in the evidence. It cannot check that a *comparison*
is true. Two false claims survived it and were found only by manual
verification against the artifacts:

- "Condition complaints in protein snacks are more than double any other
  category" — false; baked goods is fractionally higher, at 14.3% against 13.9%.
- "Dairy's taste mentions are 21% negative, the lowest of any large theme
  anywhere" — false; three categories have lower rates for the same theme, and
  several other themes are far lower.

Both numbers were correct. Both superlatives built on them were wrong. A
validator of this kind buys arithmetic anchoring, not editorial accuracy, and
treating a green check as proof that the prose is true would be exactly the
wrong lesson to draw from it. Comparative and superlative claims still need
checking by hand.

## What this cannot tell you

- **Nothing causal.** See stage 07.
- **Nothing about brands or products.** 3,438 products across 4,173 reviews is
  roughly 1.2 reviews per product. Product-level comparison is not supportable
  and is not attempted.
- **Nothing about today.** The data ends in 2012.
- **Nothing about non-reviewers.** Review corpora capture people motivated
  enough to write, which is not the customer base.
