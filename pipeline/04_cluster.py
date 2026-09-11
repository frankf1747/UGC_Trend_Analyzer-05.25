"""Stage 04 — Discover candidate themes by clustering sentence embeddings.

Clustering is global rather than per-category on purpose: a theme like "arrived
damaged" should mean the same thing in coffee and in candy, so that categories
can be compared against one another later.

On method: density clustering (HDBSCAN, the BERTopic default) was tried first and
rejected. On this corpus — 9,954 short sentences, 384 dimensions — it returned
2 clusters and left 90% of sentences as noise at every parameter setting worth
trying. Short review sentences simply do not form density peaks. Discarding 90%
of the evidence to preserve a fashionable method would have been the wrong trade.

Spherical k-means is used instead: cosine geometry (the embeddings are already
L2-normalised, so Euclidean k-means on them *is* spherical k-means), every
sentence assigned, k chosen by silhouette over a grid. It produces deliberately
over-fine micro-clusters, which stage 05 merges into a small named taxonomy.
"""
import json
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config import CORPUS, EMBEDDINGS, CLUSTERS, EXEMPLARS, RANDOM_SEED

K_GRID = [24, 32, 40, 48, 56]
N_EXEMPLARS = 10
SILHOUETTE_SAMPLE = 4000

def main() -> None:
    df = pd.read_parquet(CORPUS)
    emb = np.load(EMBEDDINGS)
    assert len(df) == len(emb), f"corpus/embedding mismatch: {len(df)} vs {len(emb)}"

    best_k, best_score, scores = None, -1.0, {}
    for k in K_GRID:
        labels = KMeans(n_clusters=k, n_init=8, random_state=RANDOM_SEED).fit_predict(emb)
        s = silhouette_score(emb, labels, metric="cosine",
                             sample_size=SILHOUETTE_SAMPLE, random_state=RANDOM_SEED)
        scores[k] = round(float(s), 4)
        print(f"  k={k:3d}  silhouette={s:.4f}", flush=True)
        if s > best_score:
            best_k, best_score = k, s

    km = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_SEED).fit(emb)
    df["cluster_id"] = km.labels_

    exemplars = {}
    for cid in range(best_k):
        idx = np.where(km.labels_ == cid)[0]
        centroid = km.cluster_centers_[cid] / np.linalg.norm(km.cluster_centers_[cid])
        order = idx[np.argsort(-(emb[idx] @ centroid))][:N_EXEMPLARS]
        sub = df.iloc[idx]
        exemplars[int(cid)] = {
            "n": int(len(idx)),
            "mean_star": round(float(sub.Score.mean()), 2),
            "top_categories": sub.analysis_category.value_counts().head(3).to_dict(),
            "sentences": df.iloc[order].sentence.str.slice(0, 200).tolist(),
        }

    df[["sent_id", "cluster_id"]].to_parquet(CLUSTERS, index=False)
    EXEMPLARS.write_text(json.dumps({"k": best_k, "silhouette_grid": scores,
                                     "clusters": exemplars}, indent=1))
    sizes = df.cluster_id.value_counts()
    print(f"stage 04 · k={best_k} (silhouette {best_score:.4f}) · "
          f"sizes {sizes.min()}–{sizes.max()}, median {int(sizes.median())}")

if __name__ == "__main__":
    main()
