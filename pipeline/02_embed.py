"""Stage 02 — Sentence embeddings with BGE-small.

Uses `transformers` directly (CLS pooling + L2 normalisation, which is what
bge-small expects) rather than adding a sentence-transformers dependency.
Runs on Apple MPS when available.
"""
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from config import CORPUS, EMBEDDINGS, EMBED_MODEL

BATCH = 64

def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

def main() -> None:
    df = pd.read_parquet(CORPUS)
    dev = device()
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL).to(dev).eval()

    out = []
    sents = df["sentence"].tolist()
    for i in range(0, len(sents), BATCH):
        batch = tok(sents[i:i + BATCH], padding=True, truncation=True,
                    max_length=256, return_tensors="pt").to(dev)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state[:, 0]      # CLS token
        out.append(torch.nn.functional.normalize(hidden, p=2, dim=1).cpu().numpy())
        if i % (BATCH * 20) == 0:
            print(f"  {i}/{len(sents)}", flush=True)

    emb = np.vstack(out).astype(np.float32)
    np.save(EMBEDDINGS, emb)
    print(f"stage 02 · {emb.shape[0]} x {emb.shape[1]} embeddings on {dev} -> {EMBEDDINGS.name}")

if __name__ == "__main__":
    main()
