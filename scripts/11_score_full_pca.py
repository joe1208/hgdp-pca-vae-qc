#!/usr/bin/env python3
# =============================================================================
# Full-chromosome (4.7M) PCA-only scorer: the same five metrics as
# 06_score_metrics.py, over directories matching ^full_config([1-6])$. Kept
# separate from the main metrics_planB.tsv; writes results/metrics_full_pca.tsv.
# Run order: sensitivity path (step 11); driven by 10_full_chrom_pca.sh.
# Inputs:    pca/full_config*/pcs.eigenvec, data/raw/...metadata.txt
# Outputs:   results/metrics_full_pca.tsv
# Analysis script, organised and commented for submission; logic unchanged.
# =============================================================================
"""
Score PCA-only full-chromosome sensitivity outputs.

This writes results/metrics_full_pca.tsv and intentionally stays separate
from metrics_planB.tsv, whose expected final shape is the 48-row main
47k/470k PCA versus VAE table.
"""
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier


from config import PROJECT_ROOT, find_metadata
META_PATH = find_metadata()
OUT_PATH = PROJECT_ROOT / "results/metrics_full_pca.tsv"
PCA_DIR_RE = re.compile(r"^full_config([1-6])$")


meta = pd.read_csv(META_PATH, sep="\t")[["sample", "region", "latitude", "longitude"]]
regions_n = meta["region"].nunique()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def score_embedding(embed_df):
    df = embed_df.merge(meta, on="sample", how="left").dropna(subset=["region"])
    x = df[["dim1", "dim2"]].values
    labels = df["region"].values

    km = KMeans(n_clusters=regions_n, n_init=100, random_state=42)
    pred = km.fit_predict(x)

    r2s = []
    for target in ["latitude", "longitude"]:
        y = df[target].values
        mask = ~np.isnan(y)
        if mask.sum() < 10:
            r2s.append(np.nan)
        else:
            model = LinearRegression().fit(x[mask], y[mask])
            r2s.append(model.score(x[mask], y[mask]))

    loo = LeaveOneOut()
    knn_preds = np.empty(len(labels), dtype=object)
    for train_idx, test_idx in loo.split(x):
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(x[train_idx], labels[train_idx])
        knn_preds[test_idx[0]] = knn.predict(x[test_idx])[0]

    return {
        "silhouette": silhouette_score(x, labels, metric="euclidean"),
        "ari": adjusted_rand_score(labels, pred),
        "geo_r2": float(np.nanmean(r2s)),
        "knn_f1": f1_score(labels, knn_preds, average="macro"),
        "calinski_harabasz": calinski_harabasz_score(x, labels),
    }


rows = []
for d in sorted((PROJECT_ROOT / "pca").glob("full_config*")):
    m = PCA_DIR_RE.match(d.name)
    if not m:
        print(f"SKIP {d.name}: not an exact full PCA directory")
        continue
    eigenvec = d / "pcs.eigenvec"
    if not eigenvec.exists():
        print(f"SKIP {d.name}: no pcs.eigenvec")
        continue

    pcs = pd.read_csv(eigenvec, sep="\t").rename(
        columns={"#IID": "sample", "PC1": "dim1", "PC2": "dim2"}
    )
    pcs = pcs[["sample", "dim1", "dim2"]]
    scores = score_embedding(pcs)
    row = {"method": "PCA", "size": "full", "config": int(m.group(1)), "seed": np.nan}
    row.update(scores)
    rows.append(row)
    print(
        f"full c{m.group(1)}: silh={scores['silhouette']:.3f} "
        f"ari={scores['ari']:.3f} geo_r2={scores['geo_r2']:.3f} "
        f"knn_f1={scores['knn_f1']:.3f}"
    )

df = pd.DataFrame(rows)
if rows:
    df = df[
        [
            "method",
            "size",
            "config",
            "seed",
            "silhouette",
            "ari",
            "geo_r2",
            "knn_f1",
            "calinski_harabasz",
        ]
    ]
df.to_csv(OUT_PATH, sep="\t", index=False)
print(f"{len(df)} rows written to {OUT_PATH}")
