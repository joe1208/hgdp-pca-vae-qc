# Running the pipeline

This documents how the analysis was run on Apocrita (QMUL's HPC) and how to run
it elsewhere. Paths are resolved through `scripts/config.py` and `scripts/env.sh`
(project root from `QC_VAE_PCA_ROOT`, else this `code_appendix` folder; raw inputs
discovered in `data/raw`), so no path edits are needed. It is still not a
one-button "run all": the VAE branch needs a GPU and Slurm, the inputs include a
~20 GB VCF, and the steps run in different places. The realistic unit of
automation is the per-(size, config) CPU chain in `run_qc_pca.sh`, with the GPU
and scoring steps submitted separately.

## Setup (once)

1. `pip install -r requirements.txt` (PLINK2 and pgenlib are not pip packages;
   see the file).
2. Put the HGDP chromosome-5 VCF (`*.vcf.gz`) and sample metadata
   (`*metadata*.txt`) in `data/raw/`.
3. Optional: `export QC_VAE_PCA_ROOT=/path/to/project` to use a project root
   outside this folder (the HPC used `/gpfs/scratch/bt24080/qc_vae_pca`).

## What runs where

| Step | Script | Where |
|---|---|---|
| 01 Subsets | `scripts/01_make_subsets.py` | HPC, CPU (once) |
| 02 Stage-1 QC | `scripts/02_stage1_qc.sh` | HPC, CPU (per size) |
| 03 Stage-2 QC | `scripts/03_stage2_qc.sh` | HPC, CPU (per size x config) |
| 04 PCA | `scripts/04_run_pca.sh` | HPC, CPU (per size x config) |
| 05 VAE | `scripts/05_train_vae.py` via `vae_slurm*.sh` | HPC, **GPU via Slurm** (36 jobs) |
| 06 Score | `scripts/06_score_metrics.py` | HPC, CPU (heavy; ran as a Slurm job) |
| 08 Seed SD | `scripts/08_vae_seed_sd.py` | CPU (reads metrics only) |
| 09 Figures | `scripts/09_build_figures.py` | HPC, CPU |
| 10-11 Full-chr PCA | `scripts/10_full_chrom_pca.sh` -> `11_score_full_pca.py` | HPC, CPU (PCA-only sensitivity) |

There is no step 07: it produced descriptive supplementary statistics that are not
used in the final write-up, so it is intentionally omitted.

## Main run order

Activate the appropriate conda env first (`conda_env` for CPU steps,
`conda_env_gpu` for VAE), and `module load plink/2.00a6LM` where PLINK2 is used.

### 1. Variant subsets (once)

```
python scripts/01_make_subsets.py
```

### 2-4. QC and PCA (per size, per config)

Either call the steps directly, or use the thin CPU-only driver, which chains
Stage-1 -> Stage-2 -> PCA for one (size, config):

```
bash run_qc_pca.sh 47k 1        # Stage-1 (47k) + Stage-2 (47k,1) + PCA (47k,1)
```

Direct equivalents:

```
bash scripts/02_stage1_qc.sh 47k                  # once per size
bash scripts/03_stage2_qc.sh 47k 1                # per size x config (12 total)
bash scripts/04_run_pca.sh   47k 1                # per size x config (12 total)
```

`run_qc_pca.sh` deliberately stops after PCA: it does not submit the VAE jobs or
run scoring.

### 5. VAE training (GPU, Slurm; 36 jobs)

Three seeds (42, 43, 44) for each of the 12 (size, config) cells. Use the bigmem
wrapper for the high-variant 470k Config 1 and Config 4; the standard wrapper for
the rest. The wrappers write Slurm logs to a relative `logs/` directory, so
submit from the project root and create it first (`mkdir -p logs`). The GPU
interpreter defaults to `$PROJECT_ROOT/conda_env_gpu/bin/python`; override with
`VAE_PYTHON` (standard wrapper) or `VAE_CONDA_ENV` (bigmem wrapper), and adjust
the `--partition`/`--account` directives for a non-Apocrita cluster.

```
# standard (32G):
sbatch --job-name=vae_47k_c1_s42 scripts/vae_slurm.sh 47k 1 42

# bigmem (64G), for 470k configs 1 and 4:
sbatch --job-name=vae_470k_c1_s42 scripts/vae_slurm_bigmem.sh 470k 1 42
```

Each job writes `vae/<size>_config<N>_seed<S>/` (latent.tsv, history.tsv,
sanity.txt, best_z_checkpoint.npy, run.json, and a per-run PNG).

### 6. Scoring

```
python scripts/06_score_metrics.py            # -> results/metrics_planB.tsv
```

(In practice this ran as a short CPU Slurm job because login-node throttling made
the bare run slow; the script itself is a plain Python program.)

### 8. Seed variability

```
python scripts/08_vae_seed_sd.py              # -> results/vae_seed_sd.tsv
```

### 9. Figures

```
python scripts/09_build_figures.py            # -> figures/*.{pdf,png}
```

## Full-chromosome PCA sensitivity (separate path)

PCA only; the VAE is out of scope at 4.7M variants (the in-RAM dosage cache
exceeds GPU memory at that scale). The driver is feasibility-gated and will not
rebuild the full dataset from the raw VCF:

```
bash scripts/10_full_chrom_pca.sh             # builds full configs (if staged),
                                              # runs PCA, then 11_score_full_pca.py
                                              # -> results/metrics_full_pca.tsv
```

## Optional plots

```
python plots/plot_pca.py 47k 1                # one config's PC1/PC2
python plots/plot_latent.py path/to/latent.tsv
python plots/plot_all_latents.py             # re-plot every VAE latent
python plots/plot_arc_geography.py           # supplementary S4: 470k cfg1 seed42 arc by lon/lat
```

## Test

Run on the HPC where the QC outputs and PLINK2 are available:

```
python tests/test_genotype_loader.py
```
