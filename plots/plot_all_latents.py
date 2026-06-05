#!/usr/bin/env python3
# =============================================================================
# Bulk latent-plot regenerator: writes a latent.png next to every
# vae/*_seed*/latent.tsv, using a fixed seven-region colour map.
# Run order: optional plotting utility (after the VAE runs).
# Inputs:    vae/*_seed*/latent.tsv, data/raw/...metadata.txt
# Outputs:   vae/*_seed*/latent.png
# Plotting script, organised and commented for submission; logic unchanged.
# =============================================================================
"""
Plot all VAE latent spaces for visual comparison with the Day-4 PNGs.
Iterates over every vae/*_seed*/latent.tsv and writes a latent.png next
to it. Old PNGs were renamed to latent_OLD.png in advance, so nothing
is overwritten destructively.

Usage:
    python plots/plot_all_latents.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on a compute node
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from config import PROJECT_ROOT, find_metadata
META_PATH = find_metadata()
VAE_DIR = PROJECT_ROOT / "vae"

# Seven continental regions, fixed colour map so configs are visually
# comparable regardless of which clusters happen to be present.
REGION_COLOURS = {
    "AFRICA":             "#2b8c5d",
    "MIDDLE_EAST":        "#d4a017",
    "EUROPE":             "#3a7ca5",
    "CENTRAL_SOUTH_ASIA": "#8b3a3a",
    "EAST_ASIA":          "#9c27b0",
    "OCEANIA":            "#e76f51",
    "AMERICA":            "#1f4e5f",
}

def main():
    meta = pd.read_csv(META_PATH, sep="\t")[["sample", "region"]]

    latent_files = sorted(VAE_DIR.glob("*_seed*/latent.tsv"))
    if not latent_files:
        print(f"No latent.tsv files found under {VAE_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(latent_files)} latent files. Plotting...")

    n_done, n_skipped = 0, 0
    for f in latent_files:
        df = pd.read_csv(f, sep="\t").merge(meta, on="sample", how="left")
        n_unmatched = df["region"].isna().sum()
        if n_unmatched:
            print(f"  WARN: {f.parent.name}: {n_unmatched}/{len(df)} samples unmatched")

        # Parse run identifier from directory name, e.g. "47k_config6_seed42"
        run_id = f.parent.name

        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=120)
        for region, colour in REGION_COLOURS.items():
            sub = df[df["region"] == region]
            if len(sub):
                ax.scatter(sub["z1"], sub["z2"], s=14, c=colour,
                           label=region.replace("_", " ").title(),
                           edgecolor="white", linewidth=0.3, alpha=0.85)
        # Catch any unlabelled samples in grey
        unmatched = df[df["region"].isna()]
        if len(unmatched):
            ax.scatter(unmatched["z1"], unmatched["z2"], s=14, c="#aaaaaa",
                       label="(no region)", edgecolor="white", linewidth=0.3, alpha=0.6)

        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_title(run_id, fontsize=11)
        ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_aspect("equal", adjustable="datalim")

        out_path = f.parent / "latent.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        n_done += 1

    print(f"\nWrote {n_done} PNGs.")

if __name__ == "__main__":
    main()