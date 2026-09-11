"""Shared paths and constants for the UGC Trend Analyzer pipeline.

Every stage reads from and writes to `artifacts/`. The pipeline is designed to
run once; the committed artifacts are what the dashboard consumes.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

SOURCE_SENTENCES = DATA / "source_sentences_2025.csv"

CORPUS = ARTIFACTS / "corpus.parquet"              # 01: cleaned sentence corpus
EMBEDDINGS = ARTIFACTS / "embeddings.npy"          # 02: sentence vectors
POLARITY = ARTIFACTS / "polarity.parquet"          # 03: model-scored polarity
CLUSTERS = ARTIFACTS / "clusters.parquet"          # 04: cluster assignments
EXEMPLARS = ARTIFACTS / "cluster_exemplars.json"   # 04: input to taxonomy naming
TAXONOMY = ARTIFACTS / "taxonomy.json"             # 05: frozen, human-reviewed labels
THEMES = ARTIFACTS / "themes.parquet"              # 06: sentences with final themes
DRIVERS = ARTIFACTS / "drivers.parquet"            # 07: rating-association effects
LANGUAGE = ARTIFACTS / "language.parquet"          # 08: weighted log-odds phrases
EVIDENCE = ARTIFACTS / "evidence_packets.json"     # 09: per-category evidence
BRIEFS = ARTIFACTS / "briefs.json"                 # 10: validated narratives
VALIDATION = ARTIFACTS / "validation.json"         # label-accuracy audit

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
POLARITY_MODEL = "siebert/sentiment-roberta-large-english"

# Analytical guards
MIN_SENTENCES_PER_CATEGORY = 150   # below this, a category is reported but not modelled
MIN_SENTENCES_PER_THEME = 30       # below this, a theme is pooled into "Other"
RANDOM_SEED = 42
