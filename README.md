# code_appendix

**This folder is the analysis code for the dissertation, organised and commented
for submission.**

The scripts have been renamed and regrouped for legibility, and their header and
docstring comments rewritten for clarity. **No analysis logic, hyperparameter,
threshold, seed, or output filename was changed. The behaviour is identical to
the code that produced the reported results.** The only non-cosmetic changes are
infrastructural: paths were made configurable (see "Implementation notes"), the
test gained a `sys.path` line so it can import the loader, and the optional PCA
plot helper gained a PC-pair selection flag. The results artefact of record is
`results/metrics_planB.tsv`.

Every script carries a header line recording that it is the organised version
with its logic unchanged, so that status is clear from the file itself.

---

## What the project does

Tests whether standard genomic quality-control steps (LD pruning, MAF filtering),
designed around the mathematical properties of PCA, are necessary or
beneficial for VAE-based population-structure inference. It compares PCA against
a VAE across six QC configurations and two dataset sizes (47k and 470k chr5
variant subsets) of the HGDP data, scoring each embedding with five metrics:
silhouette score, adjusted Rand index, geographic R2, k-NN F1, and the
Calinski-Harabasz index. A PCA-only sensitivity check is also run at full
chromosome scale (~4.7M variants).

## Pipeline overview

Two-stage QC feeds two parallel embedding branches, which are then scored and
summarised:

1. **Subsets**: draw nested 47k/470k variant subsets (47k is a strict prefix
   of 470k).
2. **Stage-1 QC**: chr5 biallelic ACGT SNPs, ID rebuild, missingness and HWE
   filters (once per dataset size).
3. **Stage-2 QC**: per-config MAF filter; configs 4-6 add LD pruning (once per
   size x config = 12 QC'd datasets).
4. **PCA branch**: PLINK2 `--pca 10` (deterministic, one run per config).
5. **VAE branch**: PopVAE-style 2D-latent VAE, three seeds (42/43/44) per
   config, trained on the GPU via Slurm.
6. **Scoring**: the five metrics for every PCA and VAE embedding.
7. **Seed variability**: per-cell VAE mean and SD across the three seeds.
8. **Figures**: the dissertation grid figures.
9. **Sensitivity path**: PCA-only full-chromosome (4.7M) analysis.

See [`RUNNING.md`](RUNNING.md) for the exact run order and which steps run
locally (CPU) versus on the cluster (Slurm/GPU).

## Directory layout

```
code_appendix/
  README.md                      this file
  RUNNING.md                     run order + "what runs where"
  requirements.txt               Python dependencies (plus PLINK2/pgenlib notes)
  run_qc_pca.sh                  thin CPU-only driver (Stage-1 -> Stage-2 -> PCA)
  data/raw/                      place the HGDP VCF (*.vcf.gz) and metadata here
  scripts/
    config.py                    path config: project root + data/raw discovery
    env.sh                       path config for the shell steps (sourced)
    01_make_subsets.py           nested 47k/470k variant subsets
    02_stage1_qc.sh              Stage-1 QC
    03_stage2_qc.sh              Stage-2 QC (MAF + optional LD pruning)
    04_run_pca.sh                PCA branch (PLINK2 --pca 10)
    05_train_vae.py              VAE branch (trainer)
    06_score_metrics.py          five metrics -> results/metrics_planB.tsv
    08_vae_seed_sd.py            per-cell VAE mean and seed SD
    09_build_figures.py          dissertation grid figures
    10_full_chrom_pca.sh         full-chromosome PCA-only driver (sensitivity)
    11_score_full_pca.py         scorer for the full-chromosome path
    genotype_loader.py           pgenlib .pgen dosage loader (used by 05)
    vae_slurm.sh                 Slurm wrapper for 05 (32G)
    vae_slurm_bigmem.sh          Slurm wrapper for 05 (64G; 470k cfg 1 & 4)
  plots/
    plot_pca.py                  single-config PC1/PC2 plot
    plot_latent.py               single latent.tsv re-plot
    plot_all_latents.py          bulk latent re-plot over all VAE runs
    plot_arc_geography.py        supplementary S4: 470k cfg1 seed42 arc by lon/lat
  tests/
    test_genotype_loader.py      loader integrity + PLINK cross-check
  results/
    metrics_planB.tsv            frozen scoring artefact of record (48 rows)
```

The main 47k/470k pipeline is `scripts/01`-`09`; the full-chromosome PCA
sensitivity path is `scripts/10`-`11`. The Slurm wrappers and the loader are
named rather than numbered because they are launchers and a library, not
sequential pipeline steps. There is no step 07: it generated descriptive
supplementary statistics that are not used in the final write-up, so it is
intentionally omitted.

## Implementation notes

- **Paths are configurable, not hardcoded.** All scripts resolve the project
  root through `scripts/config.py` (Python) and `scripts/env.sh` (shell): the
  root is taken from the `QC_VAE_PCA_ROOT` environment variable if set, and
  otherwise defaults to this `code_appendix` directory, so the pipeline runs
  from wherever it is unpacked. The original HPC runs used
  `QC_VAE_PCA_ROOT=/gpfs/scratch/bt24080/qc_vae_pca`. Raw inputs (the HGDP VCF
  and sample metadata) are discovered by pattern in `data/raw`, so the only
  setup step is to drop those two files there. Only path resolution changed;
  no analysis logic, threshold, hyperparameter, or output filename was altered.

- **`metrics_planB.tsv` filename is retained.** The scorer is named
  `06_score_metrics.py`, but it still writes `results/metrics_planB.tsv`, and the
  stats scripts still read that name. That file is the results artefact of record
  for the dissertation; renaming it would alter behaviour and break the link to
  the reported numbers, so the data filename is left unchanged.

- **Genotype loading.** `genotype_loader.py` reads PLINK 2 binary `.pgen` files
  directly via the `pgenlib` bindings, encodes genotypes as integer dosages
  (0/1/2), and mean-imputes missing genotypes per variant. Imputation happens
  before the on-disk `.npy` cache is written, and a no-NaN assertion guards the
  cached array, so a cached dosage matrix can never contain NaNs. The VAE
  reconstruction objective is categorical cross-entropy over the three dosage
  classes per variant (decoder output dimension 3 x N_variants).

- **Test import path.** `tests/test_genotype_loader.py` adds the sibling
  `scripts/` directory to `sys.path` so it can import `genotype_loader`,
  preserving the original import semantics (same module, same function).

## Setup

1. Install the Python dependencies: `pip install -r requirements.txt` (Python
   3.11). PLINK2 and `pgenlib` are not pip packages; see `requirements.txt`.
2. Put the two HGDP inputs in `data/raw/`: the chromosome-5 VCF (`*.vcf.gz`) and
   the sample metadata (`*metadata*.txt`). The scripts discover them by pattern,
   so the exact filenames do not matter.
3. Optional: `export QC_VAE_PCA_ROOT=/path/to/project` to point the pipeline at a
   project root outside this folder (the HPC used the Apocrita scratch path).

Output directories (`qc/`, `pca/`, `vae/`, `results/`, `figures/`) are created
by the scripts as needed, so no directory-setup step is required.

## Data

The dataset is the public HGDP high-coverage whole-genome release of Bergstrom
et al. 2020 (Science, "Insights into human genetic variation and population
history from 929 diverse genomes"). This analysis uses two files from the
`hgdp_wgs.20190516` release:

- the chromosome-5 VCF, `hgdp_wgs.20190516.full.chr5.vcf.gz`
- the sample metadata, `hgdp_wgs.20190516.metadata.txt`

Both files are hosted over HTTPS by the Wellcome Sanger Institute in the
`hgdp_wgs.20190516` release. The per-chromosome VCFs sit in the release root and
the metadata in a `metadata/` subfolder:

- chromosome-5 VCF:
  `https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/hgdp_wgs.20190516.full.chr5.vcf.gz`
- sample metadata:
  `https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/metadata/hgdp_wgs.20190516.metadata.txt`

The release index is browsable at
`https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/`. Download both into
`data/raw/` (Setup step 2), for example:

    cd data/raw
    wget https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/hgdp_wgs.20190516.full.chr5.vcf.gz
    wget https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/metadata/hgdp_wgs.20190516.metadata.txt

Related resources: the raw sequencing reads are under ENA study accession
PRJEB6463, and a harmonised gnomAD HGDP + 1000 Genomes callset covers the same
samples as an alternative (a re-harmonised callset, not identical to this release).

The chromosome-5 VCF is large (roughly 20 GB), so it is not included with this
code; only the small frozen results file (`results/metrics_planB.tsv`) is shipped.
From that file alone you can reproduce the seed-variability table
(`08_vae_seed_sd.py`) and the metric-comparison figure (Figure 2). The remaining
figures additionally require the sample metadata and the PCA/VAE embedding
outputs, which are not shipped due to size.

## Environment

- Python 3.11.15; two conda envs on scratch (`conda_env` CPU, `conda_env_gpu`
  GPU).
- PLINK2 v2.00a6LM (system module); `pgenlib` (PLINK 2 Python bindings).
- PyTorch 2.4.1 (CUDA 12.4); scikit-learn 1.3.2; scipy; pandas; numpy;
  matplotlib.
- HPC: Apocrita at QMUL (Slurm). CPU nodes for QC/PCA; V100-PCIE-32GB GPU nodes
  (gpushort partition) for VAE training.
- Dataset: HGDP chromosome 5 (Bergstrom et al. 2020), 929 unrelated individuals,
  7 continental regions.
