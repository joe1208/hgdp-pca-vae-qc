#!/usr/bin/env python3
# =============================================================================
# Across-seed mean and sample SD (ddof=1) of the VAE primary metrics per
# (size, config) cell. PCA is deterministic and has no seeds, so this seed
# variability summary is VAE-specific. A guardrail requires exactly three seeds
# (42/43/44) per cell before any SD is computed.
# Run order: step 08 of the main pipeline; after scoring (06).
# Inputs:    results/metrics_planB.tsv
# Outputs:   results/vae_seed_sd.tsv
# Analysis script, organised and commented for submission; logic unchanged.
# =============================================================================
"""
Across-seed mean and sample SD for VAE primary metrics, per (size, config) cell.

This is the single committed source of truth for the per-cell VAE standard
deviations reported in the Results chapter (e.g. section 3.7). Every prior SD
figure in the write-up was recomputed ad hoc, which repeatedly risked mixing
the sample (n-1) and population (n) conventions. This script fixes the
convention to sample SD (ddof=1, Bessel-corrected), because the three seeds
(42/43/44) are a sample of possible random initialisations rather than the
entire population of them.

Guardrail: every VAE cell must contain exactly three seed rows (42, 43, 44).
If any cell has more or fewer rows, the script stops before computing any SD,
because that indicates legacy or duplicate rows in metrics_planB.tsv and the
SDs would be computed over the wrong set.
"""

from datetime import datetime
import pandas as pd

from config import PROJECT_ROOT

METRICS_PATH = PROJECT_ROOT / "results" / "metrics_planB.tsv"
OUT_TSV = PROJECT_ROOT / "results" / "vae_seed_sd.tsv"

PRIMARY_METRICS = ["silhouette", "ari", "geo_r2", "knn_f1"]
EXPECTED_SEEDS = {42, 43, 44}
SD_DDOF = 1  # sample SD (n-1), Bessel-corrected

REQUIRED_COLUMNS = {"method", "size", "config", "seed", *PRIMARY_METRICS}


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}")


def load_vae_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"metrics file not found: {METRICS_PATH}")
    df = pd.read_csv(METRICS_PATH, sep="\t")
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "metrics_planB.tsv is missing required columns: "
            + ", ".join(sorted(missing))
        )
    vae = df.loc[df["method"] == "VAE"].copy()
    if vae.empty:
        raise ValueError("no VAE rows found in metrics_planB.tsv")
    vae["seed"] = vae["seed"].astype(int)
    return vae


def verify_three_seeds_per_cell(vae: pd.DataFrame) -> None:
    """Stop with a clear report if any cell is not exactly seeds {42, 43, 44}."""
    log("Row count per VAE cell (must be exactly three seeds 42/43/44):")
    problems = []
    for (size, config), group in vae.groupby(["size", "config"]):
        seeds = sorted(group["seed"].tolist())
        seed_set = set(seeds)
        flag = "" if (len(seeds) == 3 and seed_set == EXPECTED_SEEDS) else "  <-- PROBLEM"
        log(f"  size={size:<5} config={config}  n={len(seeds)}  seeds={seeds}{flag}")
        if flag:
            problems.append((size, config, seeds))

    if problems:
        log("STOP: the following cells are not exactly three seeds (42, 43, 44):")
        for size, config, seeds in problems:
            log(f"  size={size} config={config} seeds={seeds}")
        raise ValueError(
            "metrics_planB.tsv has cells that are not exactly three seeds; "
            "resolve legacy/duplicate rows before computing SDs."
        )
    log("All 12 VAE cells have exactly three seeds. Proceeding.")


def compute_table(vae: pd.DataFrame) -> pd.DataFrame:
    agg = (
        vae.groupby(["size", "config"])[PRIMARY_METRICS]
        .agg(["mean", lambda s: s.std(ddof=SD_DDOF)])
    )
    # Flatten the MultiIndex columns to <metric>_mean / <metric>_sd.
    agg.columns = [
        f"{metric}_{'mean' if stat == 'mean' else 'sd'}"
        for metric, stat in agg.columns
    ]
    table = agg.reset_index().sort_values(["size", "config"]).reset_index(drop=True)
    return table


def main() -> None:
    log(f"Reading {METRICS_PATH}")
    vae = load_vae_metrics()
    verify_three_seeds_per_cell(vae)

    table = compute_table(vae)
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_TSV, sep="\t", index=False)
    log(f"Wrote {OUT_TSV}")

    log("Per-cell VAE mean and sample SD (n-1) for primary metrics:")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
