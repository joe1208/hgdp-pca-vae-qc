#!/usr/bin/env python3
# =============================================================================
# Compute the five evaluation metrics for every PCA and VAE embedding and write
# the results table of record. Directory regexes are anchored so non-canonical
# run directories are excluded; each metric runs in a try/except so one
# degenerate embedding cannot abort the whole pass.
# Run order: step 06 of the main pipeline; after PCA (04) and all VAE runs (05).
# Inputs:    pca/*/pcs.eigenvec, vae/*/latent.tsv, data/raw/...metadata.txt
# Outputs:   results/metrics_planB.tsv  (the results artefact of record)
# Analysis script, organised and commented for submission; logic unchanged.
# =============================================================================
"""
06_score_metrics.py: compute the five evaluation metrics for every PCA and
VAE embedding and write the results table of record.

For each run the 2-D embedding (PCA eigenvectors or VAE latent coordinates) is
merged with the HGDP sample metadata, and the following are computed:
  silhouette score, adjusted Rand index, geographic R2 (latitude/longitude),
  k-NN F1 (leave-one-out), and the Calinski-Harabasz index.

Each metric is computed inside a try/except so a single degenerate embedding
cannot abort the whole pass: any metric that raises is recorded as NaN, a
warning is logged, and scoring continues to the next run. The PCA and VAE
directory regexes are anchored, so non-canonical run directories (e.g. legacy
or duplicate folders) are excluded from scoring.

The output filename results/metrics_planB.tsv is retained verbatim: it is the
results artefact of record for the dissertation, and renaming it would break
the link to the reported numbers (see the README provenance note). PCA rows
leave the seed column blank.

Usage: python scripts/06_score_metrics.py
"""
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    f1_score,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut

from config import PROJECT_ROOT, find_metadata
META_PATH = find_metadata()
OUT_PATH = PROJECT_ROOT / "results/metrics_planB.tsv"   # results artefact of record (see README)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

PCA_DIR_RE = re.compile(r"^(47k|470k)_config([1-6])$")
VAE_DIR_RE = re.compile(r"^(47k|470k)_config([1-6])_seed(42|43|44)$")

# -----------------------------------------------------------------------------
# Load metadata once
# -----------------------------------------------------------------------------
meta = pd.read_csv(META_PATH, sep="\t")[["sample", "region", "latitude", "longitude"]]
print(f"Loaded metadata: {len(meta)} samples, {meta['region'].nunique()} regions")

N_REGIONS = meta["region"].nunique()  # should be 7 for HGDP


# -----------------------------------------------------------------------------
# Metric functions
# -----------------------------------------------------------------------------
def m_silhouette(df):
    X = df[["dim1", "dim2"]].values
    labels = df["region"].values
    return silhouette_score(X, labels, metric="euclidean")


def m_ari(df):
    X = df[["dim1", "dim2"]].values
    labels_true = df["region"].values
    km = KMeans(n_clusters=N_REGIONS, n_init=100, random_state=42)
    labels_pred = km.fit_predict(X)
    return adjusted_rand_score(labels_true, labels_pred)


def m_geo_r2(df):
    X = df[["dim1", "dim2"]].values
    r2s = []
    for target in ["latitude", "longitude"]:
        y = df[target].values
        mask = ~np.isnan(y)
        if mask.sum() < 10:
            return np.nan
        model = LinearRegression().fit(X[mask], y[mask])
        r2s.append(model.score(X[mask], y[mask]))
    return float(np.mean(r2s))


def m_knn_f1(df):
    X = df[["dim1", "dim2"]].values
    labels = df["region"].values
    loo = LeaveOneOut()
    preds = np.empty(len(labels), dtype=object)
    for train_idx, test_idx in loo.split(X):
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X[train_idx], labels[train_idx])
        preds[test_idx[0]] = knn.predict(X[test_idx])[0]
    return f1_score(labels, preds, average="macro")


def m_calinski(df):
    X = df[["dim1", "dim2"]].values
    labels = df["region"].values
    return calinski_harabasz_score(X, labels)


METRICS = {
    "silhouette":        m_silhouette,
    "ari":               m_ari,
    "geo_r2":            m_geo_r2,
    "knn_f1":            m_knn_f1,
    "calinski_harabasz": m_calinski,
}


def score_embedding(embed_df, label=""):
    """Compute every metric, surviving individual failures with NaN.

    Each metric is wrapped in try/except so a degenerate embedding (e.g.
    the collapsed 470k Config 4 seed 42 with all 929 samples at one point)
    can't crash the whole pass. NaN is recorded and a warning printed.
    """
    df = embed_df.merge(meta, on="sample", how="left").dropna(subset=["region"])
    out = {}
    for name, fn in METRICS.items():
        try:
            out[name] = fn(df)
        except Exception as e:
            out[name] = np.nan
            print(f"    WARN: {name} failed for {label}: {type(e).__name__}: {e}")
    return out


# -----------------------------------------------------------------------------
# Score all PCA runs
# -----------------------------------------------------------------------------
rows = []
pca_dirs = sorted((PROJECT_ROOT / "pca").glob("*_config*"))
print(f"\nFound {len(pca_dirs)} PCA-like directories")

for d in pca_dirs:
    m = PCA_DIR_RE.match(d.name)
    if not m:
        print(f"  SKIP {d.name}: not an exact main PCA directory")
        continue
    size, config = m.group(1), m.group(2)
    eigenvec = d / "pcs.eigenvec"
    if not eigenvec.exists():
        print(f"  SKIP {d.name}: no pcs.eigenvec")
        continue
    pcs = pd.read_csv(eigenvec, sep="\t").rename(columns={"#IID": "sample", "PC1": "dim1", "PC2": "dim2"})
    pcs = pcs[["sample", "dim1", "dim2"]]
    label = f"PCA {size} c{config}"
    scores = score_embedding(pcs, label=label)
    rows.append({"method": "PCA", "size": size, "config": int(config), "seed": np.nan, **scores})
    print(f"  {label}: silh={scores['silhouette']:.3f} ari={scores['ari']:.3f} "
          f"geo_r2={scores['geo_r2']:.3f} knn_f1={scores['knn_f1']:.3f} "
          f"ch={scores['calinski_harabasz']:.0f}")


# -----------------------------------------------------------------------------
# Score all VAE runs
# -----------------------------------------------------------------------------
vae_dirs = sorted((PROJECT_ROOT / "vae").glob("*_config*_seed*"))
print(f"\nFound {len(vae_dirs)} VAE-like directories")

for d in vae_dirs:
    m = VAE_DIR_RE.match(d.name)
    if not m:
        print(f"  SKIP {d.name}: not an exact final VAE run directory")
        continue
    size, config, seed = m.group(1), m.group(2), m.group(3)
    latent = d / "latent.tsv"
    if not latent.exists():
        print(f"  SKIP {d.name}: no latent.tsv")
        continue
    z = pd.read_csv(latent, sep="\t").rename(columns={"z1": "dim1", "z2": "dim2"})
    z = z[["sample", "dim1", "dim2"]]
    label = f"VAE {size} c{config} s{seed}"
    scores = score_embedding(z, label=label)
    rows.append({"method": "VAE", "size": size, "config": int(config), "seed": int(seed), **scores})
    print(f"  {label}: silh={scores['silhouette']:.3f} ari={scores['ari']:.3f} "
          f"geo_r2={scores['geo_r2']:.3f} knn_f1={scores['knn_f1']:.3f} "
          f"ch={scores['calinski_harabasz']:.0f}")


# -----------------------------------------------------------------------------
# Save and summarise
# -----------------------------------------------------------------------------
df = pd.DataFrame(rows)
df = df[["method", "size", "config", "seed",
         "silhouette", "ari", "geo_r2", "knn_f1", "calinski_harabasz"]]
df.to_csv(OUT_PATH, sep="\t", index=False)
print(f"\n{len(df)} rows written to {OUT_PATH}")

print("\nSummary (mean over seeds for VAE, single value for PCA):")
agg = df.groupby(["method", "size", "config"]).agg(
    silh=("silhouette", "mean"),
    silh_sd=("silhouette", "std"),
    ari=("ari", "mean"),
    geo_r2=("geo_r2", "mean"),
    knn_f1=("knn_f1", "mean"),
).round(3)
print(agg.to_string())
