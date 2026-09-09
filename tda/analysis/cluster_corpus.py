"""Content clusters over a midtraining corpus, for cluster-complete removal sets.

WHY CLUSTERS. The subset-removal test (CLAUDE.md §5.4) is a GROUP counterfactual:
drop k documents at once and retrain. For removal, redundancy works the opposite
way round from selection-for-addition -- dropping one of two near-duplicates
leaves the content intact, because its twin still trains the model. So a set of
the globally top-scoring documents spreads itself across near-duplicate families
and destroys little, while a set of whole clusters evacuates content. Heo et al.
(arXiv:2605.15675) show per-document influence mis-ranks groups under exactly
this redundancy, and that the fix is to account for within-set interaction.

`removal_arm(mode="<method>cluster_<polarity>")` reads the array this writes.

Deliberately TF-IDF + MiniBatchKMeans rather than embeddings: it needs no GPU
and no model download, it is deterministic given the seed, and near-duplicate
detection in a synthetically generated single-theme corpus is a lexical problem.
The clustering is a SELECTION RULE, not a measurement, so its only requirement
is that it be fixed and reproducible.

    python -m tda.analysis.cluster_corpus --out clusters_tfidf128.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

CORPUS = "chloeli/msm-llama-pro-america"
N_CLUSTERS = 128          # ~50 documents per cluster at 6,400; median size 18
SEED = 0


def cluster(corpus: str = CORPUS, n_clusters: int = N_CLUSTERS,
            seed: int = SEED):
    import numpy as np
    from datasets import load_dataset
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    ds = load_dataset(corpus, split="train")
    vec = TfidfVectorizer(max_features=50000, sublinear_tf=True,
                          stop_words="english", ngram_range=(1, 2), min_df=3)
    x = vec.fit_transform(ds["text"])
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=10,
                         batch_size=1024).fit(x)
    lab = km.labels_.astype(np.int16)
    assert len(lab) == len(ds), (len(lab), len(ds))
    return lab, {"corpus": corpus, "n_docs": len(ds), "n_clusters": n_clusters,
                 "seed": seed, "vectorizer": "tfidf 1-2gram min_df=3 max_feat=50k",
                 "sizes": np.bincount(lab, minlength=n_clusters).tolist()}


if __name__ == "__main__":
    import json

    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="clusters_tfidf128.npy")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    a = ap.parse_args()
    lab, meta = cluster(a.corpus, a.n_clusters)
    np.save(a.out, lab)
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {a.out}  ({meta['n_docs']} docs, {a.n_clusters} clusters, "
          f"sizes {min(meta['sizes'])}..{max(meta['sizes'])})")
