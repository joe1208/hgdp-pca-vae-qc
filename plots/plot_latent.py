#!/usr/bin/env python3
# =============================================================================
# Post-hoc re-plotter for any single latent.tsv, with sv-ratio and bounding-box
# annotations in the title. Standalone utility for regenerating a latent plot
# without re-training (05_train_vae.py does its own plotting inline).
# Run order: optional plotting utility (not part of the numbered pipeline).
# Inputs:    a latent.tsv (+ metadata)
# Outputs:   a latent_<...>.png alongside the input
# Plotting script, organised and commented for submission; logic unchanged.
# =============================================================================
"""
plot_latent.py: post-hoc plotter for any latent.tsv file produced by
05_train_vae.py. Use this to (re)generate a PNG when the trainer crashed
before plotting, or to re-style figures without re-training.

Usage:
    python plot_latent.py path/to/latent.tsv [path/to/output.png]

If output path is omitted, writes a sibling file alongside latent.tsv with
a name derived from the run directory (e.g. latent_470k_C4_S42.png).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from config import find_metadata


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python plot_latent.py path/to/latent.tsv [output.png]")
        sys.exit(2)
    latent_path = Path(sys.argv[1])
    if not latent_path.exists():
        print(f"latent file not found: {latent_path}")
        sys.exit(2)

    latent_df = pd.read_csv(latent_path, sep="\t")
    if not {"sample", "z1", "z2"} <= set(latent_df.columns):
        print(f"latent.tsv must have columns sample, z1, z2; got {list(latent_df.columns)}")
        sys.exit(2)

    # Derive output path and a title from the parent directory if given a
    # standard run dir (..._configN_seedS).
    if len(sys.argv) == 3:
        out_path = Path(sys.argv[2])
    else:
        run_dir = latent_path.parent
        # parse name like 470k_config4_seed42
        name = run_dir.name
        try:
            size = name.split("_")[0]
            config = name.split("_config")[1].split("_")[0]
            seed = name.split("_seed")[1]
            out_name = f"latent_{size}_C{config}_S{seed}.png"
        except (IndexError, ValueError):
            out_name = "latent.png"
        out_path = run_dir / out_name

    title = f"VAE latent -- {latent_path.parent.name}"

    # Sanity numbers for the title
    z = latent_df[["z1", "z2"]].to_numpy()
    z1_range = z[:, 0].max() - z[:, 0].min()
    z2_range = z[:, 1].max() - z[:, 1].min()
    bbox = z1_range * z2_range
    zc = z - z.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(zc, compute_uv=False)
    sv_ratio = sv[0] / max(sv[1], 1e-12)
    title += f"\nN={len(latent_df)}  sv ratio={sv_ratio:.2f}  bbox area={bbox:.4f}"

    # Try to merge metadata for colouring
    meta = None
    region_col = sample_col = None
    try:
        meta = pd.read_csv(find_metadata(), sep="\t", low_memory=False)
        for cand in ["region", "Region", "continental_region", "continent",
                     "GeographicRegion", "geographic_region"]:
            if cand in meta.columns:
                region_col = cand
                break
        for cand in ["sample", "sampleID", "sample_id", "ID", "iid", "IID",
                     "sample_name"]:
            if cand in meta.columns:
                sample_col = cand
                break
        if region_col is None or sample_col is None:
            meta = None
    except Exception as e:
        print(f"[warn] could not load metadata: {e}")
        meta = None

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    if meta is not None:
        plotdf = latent_df.merge(
            meta[[sample_col, region_col]],
            left_on="sample", right_on=sample_col, how="left",
        )
        plotdf[region_col] = plotdf[region_col].fillna("Unknown")
        cmap = plt.get_cmap("tab10")
        for i, r in enumerate(sorted(plotdf[region_col].unique())):
            m = plotdf[region_col] == r
            ax.scatter(plotdf.loc[m, "z1"], plotdf.loc[m, "z2"],
                       c=[cmap(i % 10)], label=str(r),
                       s=22, alpha=0.85, edgecolor="none")
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
    else:
        ax.scatter(latent_df["z1"], latent_df["z2"], s=22, alpha=0.6,
                   edgecolor="none")

    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()