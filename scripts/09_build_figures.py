#!/usr/bin/env python3
# =============================================================================
# Pipeline role: Assemble the dissertation figures (region composition bar
#                chart, PCA-vs-VAE headline, VAE and PCA block grids, seed
#                stability, population composition) as PDF + PNG.
# Run order:     Step 9 of 9, after PCA (4a) and VAE (4b) outputs exist.
# Inputs:        pca/*/pcs.eigenvec, vae/*/latent.tsv, data/raw/...metadata.txt
# Outputs:       figures/{figure1_dataset_composition,figure2_metric_comparison,
#                figure3_headline_pca_vae,figure4_vae_grid,supplementary_S3_pca_grid,
#                supplementary_S1_config4_470k_seeds,
#                supplementary_S2_population_composition}.{pdf,png}
# Analysis script, organised and commented for submission; logic unchanged.
# =============================================================================
"""
09_build_figures.py: assemble the dissertation figures from raw PCA
eigenvecs and VAE latent.tsv files.

Outputs (both PDF for vector embed and PNG for preview), into figures/:
    figure1_dataset_composition            7-region horizontal sample-count bar chart
    figure2_metric_comparison              4x2 grouped-bar PCA vs VAE, four metrics x two sizes
    figure3_headline_pca_vae               2x2 PCA vs VAE on configs 1 and 4 at 470k
    figure4_vae_grid                       VAE block grid (configs 1-3, then 4-6, x 47k/470k)
    supplementary_S3_pca_grid              PCA block grid (same block layout)
    supplementary_S1_config4_470k_seeds    1x3 VAE panels (470k Config 4 seeds 42/43/44)
    supplementary_S2_population_composition  54-population horizontal bar chart (by region)

The continental-region palette is a deterministic seven-colour subset of
the Wong (2011) / seaborn-colourblind palette, mapped to regions sorted
alphabetically. The same palette is used across all three figures so a
single dissertation-wide legend would also be valid.

Run from the project root with the analysis environment active:
    python scripts/09_build_figures.py
"""

import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from config import PROJECT_ROOT, find_metadata

META_PATH    = find_metadata()
METRICS_PATH = PROJECT_ROOT / "results/metrics_planB.tsv"
PCA_DIR      = PROJECT_ROOT / "pca"
VAE_DIR      = PROJECT_ROOT / "vae"
OUT_DIR      = PROJECT_ROOT / "figures"

SIZES                 = ["47k", "470k"]
CONFIGS               = [1, 2, 3, 4, 5, 6]
# Two QC blocks for the grid figures: configs 1-3 have no LD pruning, 4-6 do.
# Showing three configs per row (instead of six) keeps each panel large enough
# to read, and groups the panels by their LD-pruning status.
BLOCKS                = [("No LD pruning", [1, 2, 3]), ("LD pruning", [4, 5, 6])]
# Headline 2x2 figure: PCA vs VAE on the two no-MAF configs at the main scale.
HEADLINE_SIZE         = "470k"
HEADLINE_CONFIGS      = [1, 4]
SUPP_SEEDS            = [42, 43, 44]
EXPECTED_REGION_COUNT = 7

# Wong (2011) / seaborn-colourblind first-seven, hardcoded so the script
# has no seaborn dependency. Order is fixed; mapping to regions is by
# alphabetical sort of region names.
PALETTE = [
    "#0173B2",  # blue
    "#DE8F05",  # orange
    "#029E73",  # green
    "#D55E00",  # vermillion
    "#CC78BC",  # purple
    "#CA9161",  # yellow-brown
    "#56B4E9",  # sky-blue
]

# Raw metadata values (lookup keys; drive alphabetical-sort palette assignment)
# map to publication-style strings shown in legends.
REGION_DISPLAY = {
    "AFRICA":             "Sub-Saharan Africa",
    "AMERICA":            "Americas",
    "CENTRAL_SOUTH_ASIA": "Central/South Asia",
    "EAST_ASIA":          "East Asia",
    "EUROPE":             "Europe",
    "MIDDLE_EAST":        "Middle East",
    "OCEANIA":            "Oceania",
}

# Expected per-region individual counts for the 929-sample analysed set, keyed
# by raw metadata region value. Used as a sanity check in the composition figure.
EXPECTED_REGION_TOTALS = {
    "AFRICA":             104,
    "AMERICA":            61,
    "CENTRAL_SOUTH_ASIA": 197,
    "EAST_ASIA":          223,
    "EUROPE":             155,
    "MIDDLE_EAST":        161,
    "OCEANIA":            28,
}
EXPECTED_GRAND_TOTAL = 929

MARKER_SIZE          = 12   # point**2 area; was 6, enlarged so point clouds read clearly
HEADLINE_MARKER_SIZE = 22   # larger still for the big 2x2 main-text figure
PNG_DPI              = 300   # was 200; sharper preview (the PDF is vector regardless)

# Method colours for the quantitative comparison figure, deliberately distinct
# from the region palette so the two figure families are not confused. PCA is a
# light grey and VAE a mid red: a large lightness gap that stays distinguishable
# in greyscale and under colour-blind simulation, not just in colour.
METHOD_COLOURS = {"PCA": "#BDBDBD", "VAE": "#C44E52"}


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_metadata():
    if not META_PATH.exists():
        die(f"metadata not found at {META_PATH}")
    meta = pd.read_csv(META_PATH, sep="\t", low_memory=False)
    for col in ("sample", "region"):
        if col not in meta.columns:
            die(
                f"metadata is missing the expected '{col}' column. "
                f"Got: {list(meta.columns)[:12]}..."
            )
    # Keep population too when the metadata provides it, for the dataset
    # composition figure. It is optional: its absence must not break the grid
    # figures, which need only sample and region.
    pop_col = next(
        (c for c in ("population", "Population", "pop") if c in meta.columns),
        None,
    )
    keep = ["sample", "region"] + ([pop_col] if pop_col else [])
    meta = meta[keep].dropna(subset=["region"])
    if pop_col and pop_col != "population":
        meta = meta.rename(columns={pop_col: "population"})
    regions = sorted(meta["region"].unique())
    if len(regions) != EXPECTED_REGION_COUNT:
        die(
            f"expected {EXPECTED_REGION_COUNT} continental regions in metadata; "
            f"got {len(regions)}: {regions}"
        )
    if len(regions) > len(PALETTE):
        die(f"palette has {len(PALETTE)} colours; need {len(regions)}")
    return meta, regions


def palette_map(regions):
    return {r: PALETTE[i] for i, r in enumerate(regions)}


def load_metrics():
    if not METRICS_PATH.exists():
        die(f"metrics file not found at {METRICS_PATH}")
    df = pd.read_csv(METRICS_PATH, sep="\t")
    n_pca = int((df["method"] == "PCA").sum())
    n_vae = int((df["method"] == "VAE").sum())
    print(f"Loaded metrics: {n_pca} PCA rows, {n_vae} VAE rows")
    return df


def load_pca(size, config):
    path = PCA_DIR / f"{size}_config{config}" / "pcs.eigenvec"
    if not path.exists():
        die(f"missing PCA eigenvec: {path}")
    df = pd.read_csv(path, sep="\t").rename(
        columns={"#IID": "sample", "PC1": "dim1", "PC2": "dim2"}
    )
    return df[["sample", "dim1", "dim2"]]


def load_vae(size, config, seed):
    path = VAE_DIR / f"{size}_config{config}_seed{seed}" / "latent.tsv"
    if not path.exists():
        die(f"missing VAE latent: {path}")
    df = pd.read_csv(path, sep="\t").rename(columns={"z1": "dim1", "z2": "dim2"})
    return df[["sample", "dim1", "dim2"]]


def merge_with_region(df, meta, label):
    out = df.merge(meta, on="sample", how="left")
    n_missing = int(out["region"].isna().sum())
    if n_missing:
        die(f"{label}: {n_missing} samples have no region in metadata")
    return out


def draw_panel(ax, df, colours, *, title=None, row_label=None,
               marker_size=MARKER_SIZE, title_fontsize=12, label_fontsize=12):
    for region, colour in colours.items():
        sub = df[df["region"] == region]
        ax.scatter(
            sub["dim1"], sub["dim2"],
            s=marker_size, c=colour, edgecolor="none", alpha=0.85,
        )
    ax.set_box_aspect(1)
    ax.tick_params(
        labelleft=False, labelbottom=False,
        left=True, bottom=True, length=3, width=0.6,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, pad=4)
    if row_label is not None:
        # Horizontal row label to the left of column 0.
        ax.set_ylabel(
            row_label, fontsize=label_fontsize, labelpad=18,
            rotation=0, ha="right", va="center",
        )


def add_region_legend(fig, colours, ncol=None, loc="lower center",
                      bbox_to_anchor=(0.5, 0.0), fontsize=9, markersize=6):
    if ncol is None:
        ncol = len(colours)
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor=c, markeredgecolor="none",
            markersize=markersize, label=REGION_DISPLAY.get(r, r),
        )
        for r, c in colours.items()
    ]
    fig.legend(
        handles=handles,
        loc=loc,
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=fontsize,
        handletextpad=0.4,
        columnspacing=1.4,
        borderaxespad=0.0,
    )


def save_both(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{name}.pdf"
    png = OUT_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=PNG_DPI)
    plt.close(fig)
    print(f"  wrote {pdf.relative_to(PROJECT_ROOT)}")
    print(f"  wrote {png.relative_to(PROJECT_ROOT)}")


def build_region_composition(name, meta, colours):
    """Region composition (main text). One horizontal bar per continental
    region, length = number of individuals, sorted by descending count and
    coloured with the shared region palette. The count is labelled at each bar
    end and the grand total is annotated. Prints per-region totals and asserts
    the analysed sample total is the expected 929. Sample metadata only; no
    metric or frozen-evidence value is touched."""
    print(f"{name}:")
    region_totals = meta.groupby("region").size()
    grand_total = int(region_totals.sum())
    print("  Per-region totals (actual vs expected):")
    for r in sorted(region_totals.index):
        actual = int(region_totals[r])
        expected = EXPECTED_REGION_TOTALS.get(r)
        flag = "OK" if expected is not None and actual == expected else "CHECK"
        print(f"    {REGION_DISPLAY.get(r, r):<20} {actual:>4}  (expected {expected}) {flag}")
    print(f"  Grand total: {grand_total}")
    assert grand_total == EXPECTED_GRAND_TOTAL, (
        f"grand total {grand_total} does not equal {EXPECTED_GRAND_TOTAL}"
    )

    ordered = region_totals.sort_values(ascending=False)
    regions_sorted = list(ordered.index)
    values = [int(ordered[r]) for r in regions_sorted]
    bar_colours = [colours[r] for r in regions_sorted]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    y = range(len(regions_sorted))
    ax.barh(list(y), values, color=bar_colours, edgecolor="none")
    ax.set_yticks(list(y))
    ax.set_yticklabels([REGION_DISPLAY.get(r, r) for r in regions_sorted], fontsize=10)
    ax.invert_yaxis()  # largest region at the top
    ax.set_xlabel("Number of individuals", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    xmax = max(values)
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.012, yi, str(v), va="center", ha="left", fontsize=9)
    ax.set_xlim(0, xmax * 1.10)
    ax.text(
        0.985, 0.04, f"Total: {grand_total} individuals",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, style="italic",
    )
    fig.subplots_adjust(left=0.24, right=0.97, top=0.96, bottom=0.13)
    save_both(fig, name)


def build_population_composition(name, meta, colours):
    """Population composition (appendix). One horizontal bar per HGDP
    population, length = number of individuals, grouped by region (regions in
    descending total) and ordered by descending count within each region,
    coloured with the shared region palette. Population names are left-side row
    labels and counts are labelled at each bar end. Prints per-population counts
    and asserts the analysed sample total is the expected 929."""
    print(f"{name}:")
    if "population" not in meta.columns:
        die(
            "metadata has no population column, so the population composition "
            "figure cannot be built. Add a 'population' column or adjust the "
            "candidate names in load_metadata()."
        )
    counts = meta.groupby(["region", "population"]).size().reset_index(name="count")
    grand_total = int(counts["count"].sum())

    # Region blocks ordered by descending region total; descending count within.
    region_order = list(
        counts.groupby("region")["count"].sum().sort_values(ascending=False).index
    )
    rank = {r: i for i, r in enumerate(region_order)}
    ordered = counts.assign(_rank=counts["region"].map(rank)).sort_values(
        ["_rank", "count"], ascending=[True, False]
    ).reset_index(drop=True)

    print("  Per-population counts (region, population, count):")
    for _, row in ordered.iterrows():
        region_name = REGION_DISPLAY.get(row["region"], row["region"])
        print(f"    {region_name:<20}\t{row['population']:<24}\t{int(row['count'])}")
    print(f"  Grand total: {grand_total}")
    assert grand_total == EXPECTED_GRAND_TOTAL, (
        f"grand total {grand_total} does not equal {EXPECTED_GRAND_TOTAL}"
    )

    n = len(ordered)
    height = 1.6 + 0.23 * n  # portrait; scales with the number of populations
    bar_colours = [colours[r] for r in ordered["region"]]
    fig, ax = plt.subplots(figsize=(8.0, height))
    y = range(n)
    ax.barh(list(y), ordered["count"], color=bar_colours, edgecolor="none")
    ax.set_yticks(list(y))
    ax.set_yticklabels(ordered["population"], fontsize=11)
    ax.invert_yaxis()  # first population in the ordering at the top
    ax.set_xlabel("Number of individuals", fontsize=13)
    ax.tick_params(axis="x", labelsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    xmax = int(ordered["count"].max())
    for yi, v in zip(y, ordered["count"]):
        ax.text(v + xmax * 0.012, yi, str(int(v)), va="center", ha="left", fontsize=9)
    ax.set_xlim(0, xmax * 1.10)
    add_region_legend(fig, colours, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.995), fontsize=11, markersize=8)
    fig.subplots_adjust(left=0.20, right=0.96, top=1.0 - (0.7 / height), bottom=0.04)
    save_both(fig, name)


def build_block_grid(name, loader, meta, colours, label_fmt):
    """Two stacked 2x3 blocks. The top block is configs 1-3 (no LD pruning),
    the bottom block configs 4-6 (LD pruning); within each block the two rows
    are the 47k and 470k subsets. Three columns per row keep panels readable,
    and each block carries its LD-pruning label as a subfigure heading."""
    print(f"{name}:")
    fig = plt.figure(figsize=(12.5, 16.0))
    subfigs = fig.subfigures(nrows=2, ncols=1, hspace=0.06)
    for b, (block_label, configs) in enumerate(BLOCKS):
        sf = subfigs[b]
        sf.suptitle(block_label, fontsize=18, fontweight="bold")
        axes = sf.subplots(
            nrows=2, ncols=3,
            gridspec_kw={
                # top lowered to 0.85 so the block heading clears the Config
                # titles of the top row below it.
                "left": 0.12, "right": 0.99, "top": 0.85, "bottom": 0.13,
                "hspace": 0.22, "wspace": 0.14,
            },
        )
        for s, size in enumerate(SIZES):
            for c, config in enumerate(configs):
                df = merge_with_region(
                    loader(size, config),
                    meta,
                    label_fmt(size, config),
                )
                draw_panel(
                    axes[s, c], df, colours,
                    title=f"Config {config}" if s == 0 else None,
                    row_label=size if c == 0 else None,
                    marker_size=18, title_fontsize=16, label_fontsize=16,
                )
    add_region_legend(
        fig, colours, ncol=4, loc="lower center",
        bbox_to_anchor=(0.5, 0.005), fontsize=14, markersize=12,
    )
    save_both(fig, name)


def build_headline(name, meta, colours):
    """Main-text 2x2: PCA (top row) vs VAE (bottom row) on the two no-MAF
    configurations (columns: config 1 and config 4) at the main 470k scale.
    Reading down a column shows both methods on identical input, which is the
    central comparison: PCA collapses while the VAE resolves the regions."""
    print(f"{name}:")
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(8.0, 8.8),
        gridspec_kw={"hspace": 0.26, "wspace": 0.20},
    )
    # Per row: method, loader, x-axis name, y-axis name. PCA axes are the first
    # two principal components; VAE axes are the two latent dimensions.
    rows = [
        ("PCA", lambda c: load_pca(HEADLINE_SIZE, c), "PC1", "PC2"),
        ("VAE", lambda c: load_vae(HEADLINE_SIZE, c, 42), "Latent 1", "Latent 2"),
    ]
    for r, (method_label, loader, xname, yname) in enumerate(rows):
        for ci, config in enumerate(HEADLINE_CONFIGS):
            ax = axes[r, ci]
            df = merge_with_region(
                loader(config),
                meta,
                f"{method_label} {HEADLINE_SIZE} config{config}",
            )
            draw_panel(
                ax, df, colours,
                title=f"Config {config}" if r == 0 else None,
                marker_size=HEADLINE_MARKER_SIZE, title_fontsize=15,
            )
            ax.set_xlabel(xname, fontsize=13)
            if ci == 0:
                ax.set_ylabel(yname, fontsize=13)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.13)
    # Bold method label down the far left of each row (PCA top, VAE bottom).
    for r, (method_label, *_rest) in enumerate(rows):
        pos = axes[r, 0].get_position()
        fig.text(
            0.025, pos.y0 + pos.height / 2, method_label,
            rotation=90, va="center", ha="center",
            fontsize=15, fontweight="bold",
        )
    add_region_legend(
        fig, colours, ncol=4, loc="lower center",
        bbox_to_anchor=(0.5, 0.0), fontsize=11, markersize=10,
    )
    save_both(fig, name)


def build_metric_comparison(name, df):
    """Quantitative comparison (main text). Four rows (the primary metrics) by
    two columns (subset sizes). In each panel, grouped bars over configs 1-6:
    PCA (deterministic, no error bar) against VAE (mean of three seeds with an
    SD error bar). Two fixed method colours, one shared legend. The geo R2 row
    autoscales; the other three share a 0-1 axis. Prints per-cell PCA value and
    VAE mean and SD per metric for cross-checking against Table 3."""
    print(f"{name}:")
    metrics = [
        ("silhouette", "Silhouette"), ("ari", "ARI"),
        ("geo_r2", "Geographic R2"), ("knn_f1", "k-NN F1"),
    ]
    sizes = ["47k", "470k"]
    configs = [1, 2, 3, 4, 5, 6]
    mcols = [m for m, _ in metrics]
    pidx = df[df["method"] == "PCA"].set_index(["size", "config"])
    vg = df[df["method"] == "VAE"].groupby(["size", "config"])[mcols]
    vmean, vsd = vg.mean(), vg.std(ddof=1)

    print("  Per (size, config): PCA value, VAE mean (SD) per metric")
    for s in sizes:
        for c in configs:
            parts = []
            for mk, _ in metrics:
                parts.append(
                    f"{mk}=PCA {pidx.loc[(s,c), mk]:.3f}/VAE "
                    f"{vmean.loc[(s,c), mk]:.3f}({vsd.loc[(s,c), mk]:.3f})"
                )
            print(f"    {s} c{c}: " + "  ".join(parts))

    fig, axes = plt.subplots(
        nrows=4, ncols=2, figsize=(8.5, 10.5),
        gridspec_kw={"hspace": 0.30, "wspace": 0.18},
    )
    width = 0.38
    xpos = list(range(1, 7))
    for ri, (mk, mlabel) in enumerate(metrics):
        for ci, s in enumerate(sizes):
            ax = axes[ri, ci]
            pv = [pidx.loc[(s, c), mk] for c in configs]
            mv = [vmean.loc[(s, c), mk] for c in configs]
            sv = [vsd.loc[(s, c), mk] for c in configs]
            ax.bar([x - width / 2 for x in xpos], pv, width,
                   color=METHOD_COLOURS["PCA"], label="PCA")
            ax.bar([x + width / 2 for x in xpos], mv, width, yerr=sv,
                   color=METHOD_COLOURS["VAE"], label="VAE",
                   error_kw={"elinewidth": 0.8, "capsize": 2})
            ax.set_xticks(xpos)
            if ri == len(metrics) - 1:
                ax.set_xticklabels(configs, fontsize=9)
                ax.set_xlabel("QC configuration", fontsize=10)
            else:
                ax.set_xticklabels([])
            if ci == 0:
                ax.set_ylabel(mlabel, fontsize=11)
            if ri == 0:
                ax.set_title(s, fontsize=12)
            if mk != "geo_r2":
                ax.set_ylim(0, 1)
            ax.tick_params(labelsize=8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    handles = [
        Line2D([0], [0], marker="s", linestyle="", markeredgecolor="none",
               markerfacecolor=METHOD_COLOURS["PCA"], markersize=9, label="PCA"),
        Line2D([0], [0], marker="s", linestyle="", markeredgecolor="none",
               markerfacecolor=METHOD_COLOURS["VAE"], markersize=9,
               label="VAE (mean, error bars SD)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.995), fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.07)
    save_both(fig, name)


def build_supplementary(name, meta, colours):
    print(f"{name}:")
    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(8.0, 3.4),
        gridspec_kw={"wspace": 0.15},
    )
    for i, seed in enumerate(SUPP_SEEDS):
        df = merge_with_region(
            load_vae("470k", 4, seed),
            meta,
            f"VAE 470k config4 seed{seed}",
        )
        draw_panel(
            axes[i], df, colours,
            title=f"seed {seed}",
        )
    fig.subplots_adjust(left=0.04, right=0.99, top=0.90, bottom=0.30)
    add_region_legend(fig, colours, ncol=4)
    save_both(fig, name)


def main():
    meta, regions = load_metadata()
    colours = palette_map(regions)
    print(f"Regions ({len(regions)}): {regions}")
    print(f"Palette assignment: " + ", ".join(f"{r}={colours[r]}" for r in regions))
    print(f"Output dir: {OUT_DIR}")
    print()

    build_region_composition(
        "figure1_dataset_composition",
        meta,
        colours,
    )
    build_metric_comparison(
        "figure2_metric_comparison",
        load_metrics(),
    )
    build_headline(
        "figure3_headline_pca_vae",
        meta,
        colours,
    )
    build_block_grid(
        "figure4_vae_grid",
        lambda s, c: load_vae(s, c, 42),
        meta,
        colours,
        lambda s, c: f"VAE {s} config{c} seed42",
    )
    build_block_grid(
        "supplementary_S3_pca_grid",
        load_pca,
        meta,
        colours,
        lambda s, c: f"PCA {s} config{c}",
    )
    build_supplementary(
        "supplementary_S1_config4_470k_seeds",
        meta,
        colours,
    )
    build_population_composition(
        "supplementary_S2_population_composition",
        meta,
        colours,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
