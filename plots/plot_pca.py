#!/usr/bin/env python3
# =============================================================================
# Plot a chosen principal-component pair (default PC1 vs PC2) for one (size,
# config), coloured by HGDP continental region. The pair is selectable with
# --pcs, so any pair already present in pcs.eigenvec can be re-plotted from
# frozen output without recomputation (PLINK2 --pca 10 writes PC1..PC10).
# Run order: optional plotting utility (not part of the numbered pipeline).
# Inputs:    pca/<size>_config<N>/pcs.eigenvec, data/raw/...metadata.txt
# Outputs:   pca/<size>_config<N>/pc<i>_pc<j>.png
# Plotting script for submission; PC-pair selection added, plotting unchanged.
# =============================================================================
"""
Plot a principal-component pair coloured by HGDP continental region.

Usage: python plots/plot_pca.py <size> <config> [--pcs I J]
  e.g. python plots/plot_pca.py 47k 1             # PC1 vs PC2 (default)
       python plots/plot_pca.py 470k 1 --pcs 3 4  # PC3 vs PC4
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display on compute node
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from config import PROJECT_ROOT, find_metadata


def plot_pca(size, config, pcs=(1, 2), project_root=PROJECT_ROOT):
    """Re-plot one (size, config) PCA as the chosen PC pair, coloured by region.
    pcs is a (i, j) pair of 1-based component numbers present in pcs.eigenvec."""
    pcx, pcy = f"PC{pcs[0]}", f"PC{pcs[1]}"
    eigenvec_path = project_root / f"pca/{size}_config{config}/pcs.eigenvec"
    metadata_path = find_metadata()
    out_path = project_root / f"pca/{size}_config{config}/pc{pcs[0]}_pc{pcs[1]}.png"

    # Load PCA. Header is "#IID\tPC1\t...\tPC10".
    pcs_df = pd.read_csv(eigenvec_path, sep="\t").rename(columns={"#IID": "sample"})
    for col in (pcx, pcy):
        if col not in pcs_df.columns:
            raise ValueError(
                f"{col} not present in {eigenvec_path}; have {list(pcs_df.columns)}"
            )

    # Load metadata and merge.
    meta = pd.read_csv(metadata_path, sep="\t")[["sample", "region", "population"]]
    df = pcs_df.merge(meta, on="sample", how="left")
    n_unmerged = int(df["region"].isna().sum())
    if n_unmerged > 0:
        print(f"WARNING: {n_unmerged} samples with no metadata match")

    # Plot.
    regions = sorted(df["region"].dropna().unique())
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")
    for i, region in enumerate(regions):
        sub = df[df["region"] == region]
        ax.scatter(sub[pcx], sub[pcy],
                   s=20, alpha=0.7, color=cmap(i),
                   label=f"{region} (n={len(sub)})")

    ax.set_xlabel(pcx)
    ax.set_ylabel(pcy)
    ax.set_title(
        f"HGDP chr5 - {size} config {config}\n"
        f"{pcx} vs {pcy} coloured by continental region"
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Samples plotted: {len(df)}; regions: {len(regions)}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot a PCA component pair coloured by HGDP region."
    )
    parser.add_argument("size", help="dataset size, e.g. 47k or 470k")
    parser.add_argument("config", help="QC configuration number, 1-6")
    parser.add_argument(
        "--pcs", nargs=2, type=int, default=[1, 2], metavar=("I", "J"),
        help="1-based PC pair to plot (default: 1 2)",
    )
    args = parser.parse_args()
    plot_pca(args.size, args.config, pcs=tuple(args.pcs))


if __name__ == "__main__":
    main()
