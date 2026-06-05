#!/usr/bin/env python3
# =============================================================================
# Supplementary figure: re-plot the existing 470k Config 1 VAE latents (the
# clean no-MAF arc) for all three training seeds (42, 43, 44), colouring the
# same points by each individual's great-circle (haversine) distance from an
# East-African origin (Addis Ababa, 9.03N 38.74E, the origin used by
# Ramachandran et al. 2005) on a continuous colour-blind-safe scale. A smooth
# gradient along each arc, reproduced across all three seeds, shows the
# principal latent axis recovers the out-of-Africa expansion gradient
# (isolation by distance / serial founder effect) as a stable property of the
# embedding rather than a single-seed artefact. Distance is the appropriate
# axis: the arc orders populations by expansion distance from Africa, not by
# longitude (raw longitude wraps across the Pacific, placing the Americas next
# to East Asia despite their genetic adjacency). Uses existing outputs only,
# no retraining, no metric recomputation. Prints the Pearson correlation
# between position along each arc (first latent axis) and distance from Africa.
# Run order: optional supplementary figure (not part of the numbered pipeline).
# Inputs:    vae/470k_config1_seed{42,43,44}/latent.tsv,
#            data/raw/hgdp_wgs.20190516.metadata.txt (latitude/longitude, the
#            same per-individual coordinates used by the geographic-R2 metric).
# Outputs:   figures/supplementary_S4_arc_geography.{pdf,png}
# New plotting script for submission; uses existing outputs, no retraining.
# =============================================================================
"""
plot_arc_geography.py: supplementary figure demonstrating that the no-MAF VAE
arc recovers the out-of-Africa expansion gradient, reproducibly across seeds.

Takes the existing VAE latent embeddings for 470k Config 1 under all three
training seeds (42, 43, 44) and re-plots the same points coloured by each
individual's great-circle distance from an East-African origin (Addis Ababa),
on a perceptually-uniform, colour-blind-safe viridis scale shared across the
three panels. Per-individual latitude/longitude are read straight from the HGDP
metadata, the same columns the geographic-R2 metric regresses the embedding onto;
distance is derived from them with the haversine formula (no new measurements).

Run from the project root with the analysis environment active:
    python plots/plot_arc_geography.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from config import PROJECT_ROOT, find_metadata
META_PATH    = find_metadata()
VAE_DIR      = PROJECT_ROOT / "vae"
OUT_DIR      = PROJECT_ROOT / "figures"

# The clean no-MAF arc: 470k Config 1, all three seeds.
SIZE     = "470k"
CONFIG   = 1
SEEDS    = [42, 43, 44]
OUT_NAME = "supplementary_S4_arc_geography"

# East-African origin of the serial founder expansion (Ramachandran et al. 2005).
ORIGIN_NAME = "Addis Ababa"
ORIGIN_LAT  = 9.03
ORIGIN_LON  = 38.74
EARTH_RADIUS_KM = 6371.0

CMAP        = "viridis"   # perceptually uniform and colour-blind-safe
MARKER_SIZE = 16          # slightly smaller than the headline; three panels
PNG_DPI     = 300         # matches the other figures (the PDF is vector regardless)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points given in decimal degrees.
    Vectorised over numpy arrays for the first pair."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def load_coords(meta_path):
    """Per-individual latitude/longitude plus population, with distance from the
    East-African origin derived by the haversine formula."""
    meta = pd.read_csv(meta_path, sep="\t")[["sample", "population", "latitude", "longitude"]]
    meta["dist_africa"] = haversine_km(
        meta["latitude"].to_numpy(), meta["longitude"].to_numpy(),
        ORIGIN_LAT, ORIGIN_LON,
    )
    return meta


def load_seed(latent_path, coords):
    """Merge one seed's existing latent coordinates with the per-individual
    distance. dim1 (the first latent axis) is treated as position along the arc."""
    z = pd.read_csv(latent_path, sep="\t").rename(columns={"z1": "dim1", "z2": "dim2"})
    df = z.merge(coords, on="sample", how="left")
    return df


def correlations(df):
    """Pearson r between the first latent axis and distance from the origin, at
    the individual level and averaged within each population. Signs are arbitrary
    (latent-axis orientation is arbitrary); magnitudes are what is quoted."""
    sub = df.dropna(subset=["dist_africa"])
    r_ind = float(np.corrcoef(sub["dim1"].to_numpy(), sub["dist_africa"].to_numpy())[0, 1])
    g = sub.groupby("population").agg(d1=("dim1", "mean"), dd=("dist_africa", "mean"))
    r_pop = float(np.corrcoef(g["d1"].to_numpy(), g["dd"].to_numpy())[0, 1])
    return r_ind, r_pop, len(sub), len(g)


def make_figure(latent_paths, meta_path, out_dir):
    """Build the three-panel (one per seed) distance-from-Africa figure with a
    shared colour scale, and report the arc-vs-distance correlations per seed."""
    coords = load_coords(meta_path)
    frames = {seed: load_seed(path, coords) for seed, path in latent_paths.items()}

    # Shared colour scale across panels (distance is identical across seeds).
    all_dist = coords["dist_africa"].dropna().to_numpy() / 1000.0
    norm = Normalize(vmin=float(all_dist.min()), vmax=float(all_dist.max()))

    fig, axes = plt.subplots(nrows=1, ncols=len(SEEDS), figsize=(11.5, 4.2))
    fig.suptitle(
        f"VAE latent, {SIZE} Config {CONFIG} (no MAF filter), "
        f"coloured by distance from {ORIGIN_NAME}",
        fontsize=12,
    )
    sc = None
    for ax, seed in zip(axes, SEEDS):
        sub = frames[seed].dropna(subset=["dist_africa"])
        sc = ax.scatter(
            sub["dim1"], sub["dim2"],
            c=sub["dist_africa"] / 1000.0, cmap=CMAP, norm=norm,
            s=MARKER_SIZE, edgecolor="none", alpha=0.9,
        )
        ax.set_box_aspect(1)
        ax.tick_params(
            labelleft=False, labelbottom=False,
            left=True, bottom=True, length=3, width=0.6,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
        ax.set_title(f"seed {seed}", fontsize=11, pad=4)
        ax.set_xlabel("Latent 1", fontsize=10)
        if seed == SEEDS[0]:
            ax.set_ylabel("Latent 2", fontsize=10)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(f"Distance from {ORIGIN_NAME} (1000 km)", fontsize=10)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.86, bottom=0.10, wspace=0.12)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{OUT_NAME}.pdf"
    png = out_dir / f"{OUT_NAME}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=PNG_DPI)
    plt.close(fig)

    print("Pearson |r| between first latent axis and distance from "
          f"{ORIGIN_NAME}, per seed:")
    for seed in SEEDS:
        r_ind, r_pop, n_ind, n_pop = correlations(frames[seed])
        print(f"  seed {seed}: individual |r| = {abs(r_ind):.3f} (n={n_ind}), "
              f"population |r| = {abs(r_pop):.3f} (n={n_pop})")
    print(f"  wrote {pdf}")
    print(f"  wrote {png}")


def main():
    latent_paths = {}
    for seed in SEEDS:
        p = VAE_DIR / f"{SIZE}_config{CONFIG}_seed{seed}" / "latent.tsv"
        if not p.exists():
            print(f"ERROR: missing VAE latent: {p}", file=sys.stderr)
            sys.exit(1)
        latent_paths[seed] = p
    if not META_PATH.exists():
        print(f"ERROR: missing metadata: {META_PATH}", file=sys.stderr)
        sys.exit(1)
    make_figure(latent_paths, META_PATH, OUT_DIR)


if __name__ == "__main__":
    main()
