#!/usr/bin/env python3
# =============================================================================
# VAE trainer (PopVAE-style 128-64-32 encoder, 2D latent, KL warm-up 30 epochs /
# linear ramp 50, early-stopping patience 50). Mean-imputes genotypes and caches
# the dosage matrix before training. Launched on the GPU via vae_slurm.sh (or
# vae_slurm_bigmem.sh for the high-variant 470k configs); three seeds (42/43/44)
# per config. The KL-annealing rationale is documented in the docstring below.
# Run order: step 05 of the main pipeline (VAE branch); after step 03.
# Inputs:    qc/<size>_config<N>/qcd.*, data/raw/hgdp_wgs.20190516.metadata.txt
# Outputs:   vae/<size>_config<N>_seed<S>/{latent.tsv,latent.png,history.tsv,
#            sanity.txt,best_z_checkpoint.npy,run.json}
# Analysis script, organised and commented for submission; logic unchanged.
# =============================================================================
"""
05_train_vae.py: consolidated VAE trainer for the QC x method comparison.

Usage
-----
    python 05_train_vae.py SIZE CONFIG SEED

    SIZE in {47k, 470k}; CONFIG in {1..6}; SEED in {42, 43, 44}.

What this produces
------------------
    vae/{size}_config{N}_seed{S}/latent.tsv   : sample, z1, z2 per individual
    vae/{size}_config{N}_seed{S}/latent.png   : 2D scatter coloured by region
    vae/{size}_config{N}_seed{S}/history.tsv  : per-epoch train/val loss + lr
    vae/{size}_config{N}_seed{S}/sanity.txt   : collapse-detection report
    vae/{size}_config{N}_seed{S}/run.json     : Plan B run provenance

Design choices and where they came from
---------------------------------------
Locked from the Methods spec (do not change without amending Methods Sec.2.4.2):
  * encoder funnel 128 -> 64 -> 32 with ELU; mirrored decoder
  * 2-D latent (mu, log-var heads off the final encoder layer)
  * categorical cross-entropy on integer dosage targets {0, 1, 2}
  * beta = 1.0
  * Adam, lr 1e-3, batch 32, max 500 epochs, early-stopping patience 50
  * 90/10 train/val split with fixed random_state across model seeds

Plan A trainer changes (relative to the broken Day-5 script):
  * Gradient clipping raised from max_norm=1.0 to max_norm=50.0. The old
    threshold throttled every step on Configs 1 and 4 (the high-variant-count
    no-MAF configs) because sum-reduction CE produces gradient norms in the
    hundreds to thousands; clipping to 1.0 effectively ran training at 1/100
    to 1/1000 of the nominal learning rate. 50.0 is a cosmetic safety net
    that only catches genuine divergences.
  * ReduceLROnPlateau added (factor=0.5, patience=12, min_lr=1e-5). Matches
    PopVAE's patience/4 schedule. Without it, training oscillates around bad
    local minima on hard configs and early-stops on collapsed latents.
  * In-script collapse-detection sanity checks: per-dim std, singular-value
    ratio, pairwise distance distribution. Latent plot saved every run.

Plan B trainer changes (relative to Plan A; addresses 470k posterior collapse):
  * Linear KL annealing schedule. beta = 0 for the first WARMUP_EPOCHS
    (pure autoencoder; encoder is forced to learn meaningful representations
    because there is no KL pressure to ignore z). beta then ramps linearly
    from 0 to 1 over RAMP_EPOCHS. After warm-up + ramp, beta = 1 (matching
    the locked Methods value). This addresses scale-dependent posterior
    collapse on no-MAF configs at high variant counts, where the decoder
    finds it can reconstruct rare-variant data without using the latent and
    the encoder gives up. Ref: Bowman et al. 2016 ("Generating Sentences
    from a Continuous Space"), Sonderby et al. 2016 (Ladder VAEs).
  * Best-val tracking and early stopping gated until after the ramp
    completes. Val_loss during warm-up and ramp is non-comparable across
    epochs (beta is changing), so early-stopping on it would fire spuriously.
    Minimum guaranteed training length: WARMUP_EPOCHS + RAMP_EPOCHS epochs.
  * Tighter sanity checks: lowered sv ratio warning to >5; added latent
    bounding-box area check (catches all-axis collapse like 470k Config 4
    that the rank-based checks missed).
"""

import sys
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Headless matplotlib for compute nodes (no DISPLAY)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
if len(sys.argv) != 4:
    print("Usage: python 05_train_vae.py SIZE CONFIG SEED")
    print("  SIZE in {47k, 470k}; CONFIG in {1..6}; SEED an integer (e.g. 42)")
    sys.exit(2)

SIZE = sys.argv[1]
CONFIG = sys.argv[2]
try:
    SEED = int(sys.argv[3])
except ValueError:
    print(f"SEED must be an integer; got {sys.argv[3]!r}")
    sys.exit(2)

if SIZE not in {"47k", "470k"}:
    print(f"SIZE must be '47k' or '470k'; got {SIZE!r}")
    sys.exit(2)
if CONFIG not in {"1", "2", "3", "4", "5", "6"}:
    print(f"CONFIG must be one of 1..6; got {CONFIG!r}")
    sys.exit(2)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from config import PROJECT_ROOT, find_metadata
PLINK_PREFIX  = PROJECT_ROOT / f"qc/{SIZE}_config{CONFIG}/qcd"
CACHE_PATH    = PROJECT_ROOT / f"vae/{SIZE}_config{CONFIG}_dosage.npy"
OUT_DIR       = PROJECT_ROOT / f"vae/{SIZE}_config{CONFIG}_seed{SEED}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Hyperparameters (locked per Methods Sec.2.4.2 unless flagged Plan A)
# ---------------------------------------------------------------------------
LATENT_DIM       = 2
HIDDEN           = [128, 64, 32]   # encoder; decoder mirrors
LR               = 1e-3
BATCH_SIZE       = 32
MAX_EPOCHS       = 500
PATIENCE         = 50              # early-stopping patience (post-ramp only)
LR_PATIENCE      = 12              # ReduceLROnPlateau (Plan A)
LR_FACTOR        = 0.5
LR_MIN           = 1e-5
BETA_FINAL       = 1.0             # final beta after annealing (Methods spec)
WARMUP_EPOCHS    = 30              # Plan B: pure autoencoder (beta=0)
RAMP_EPOCHS      = 50              # Plan B: linear ramp 0 -> BETA_FINAL
DATA_SPLIT_SEED  = 42              # fixed across model seeds
GRAD_CLIP_NORM   = 50.0            # Plan A: was 1.0


def get_beta(epoch: int) -> float:
    """Linear KL-annealing schedule.

    Epoch 0..WARMUP_EPOCHS-1:                       beta = 0   (pure AE)
    Epoch WARMUP_EPOCHS..WARMUP_EPOCHS+RAMP-1:      beta = linear ramp 0 -> 1
    Epoch WARMUP_EPOCHS+RAMP_EPOCHS..MAX_EPOCHS:    beta = BETA_FINAL
    """
    if epoch < WARMUP_EPOCHS:
        return 0.0
    if epoch < WARMUP_EPOCHS + RAMP_EPOCHS:
        progress = (epoch - WARMUP_EPOCHS) / RAMP_EPOCHS
        return BETA_FINAL * progress
    return BETA_FINAL


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[init] device={device}, size={SIZE}, config={CONFIG}, seed={SEED}")


# ---------------------------------------------------------------------------
# Load genotypes
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from genotype_loader import load_genotypes  # noqa: E402

print(f"[load] {PLINK_PREFIX}")
geno_float, sample_list, _ = load_genotypes(
    plink_prefix=str(PLINK_PREFIX),
    cache_path=str(CACHE_PATH),
)
sample_ids = np.array(sample_list)
n_samples, n_variants = geno_float.shape
print(f"[load] {n_samples} samples x {n_variants} variants")
assert not np.isnan(geno_float).any(), "loader returned NaNs (should be impossible)"

# Round mean-imputed values to integer classes for cross-entropy targets
geno_int = np.clip(np.rint(geno_float).astype(np.int64), 0, 2)


# ---------------------------------------------------------------------------
# Train / validation split (90 / 10, fixed across model seeds)
# ---------------------------------------------------------------------------
train_idx, val_idx = train_test_split(
    np.arange(n_samples), test_size=0.1, random_state=DATA_SPLIT_SEED
)
X_train = torch.from_numpy(geno_int[train_idx]).float().to(device)
Y_train = torch.from_numpy(geno_int[train_idx]).long().to(device)
X_val   = torch.from_numpy(geno_int[val_idx]).float().to(device)
Y_val   = torch.from_numpy(geno_int[val_idx]).long().to(device)
X_full  = torch.from_numpy(geno_int).float().to(device)
print(f"[split] train n={len(train_idx)}, val n={len(val_idx)}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, n_v: int, hidden, latent: int):
        super().__init__()
        # Encoder: funnel with ELU after each linear
        enc, prev = [], n_v
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(prev, latent)
        self.fc_lv = nn.Linear(prev, latent)
        # Decoder: mirror the encoder shape, terminating at n_v * 3 logits
        dec, prev = [], latent
        for h in reversed(hidden):
            dec += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        dec += [nn.Linear(prev, n_v * 3)]
        self.decoder = nn.Sequential(*dec)
        self.n_v = n_v

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_lv(h)

    def reparameterize(self, mu, lv):
        std = torch.exp(0.5 * lv)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        out = self.decoder(z).view(-1, self.n_v, 3)
        return out, mu, lv


model = VAE(n_variants, HIDDEN, LATENT_DIM).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"[model] {n_params:,} params")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN
)


def vae_loss(out, target, mu, lv, beta):
    """Sum-reduction CE over (batch, variants) + beta * KL over (batch, latent).

    Beta is passed in per-call so the training loop can anneal it. Returns
    the total loss alongside its two components for logging.
    """
    # out is (B, V, 3); cross_entropy wants (B, 3, V) for class dim at pos 1
    recon = F.cross_entropy(out.permute(0, 2, 1), target, reduction="sum")
    kld   = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
    return recon + beta * kld, recon, kld


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
# Plan B: best_val tracking and early stopping are gated until after the KL
# ramp completes. During warm-up (beta=0) and ramp (beta growing 0->1), the
# val_loss values aren't comparable across epochs because beta is changing,
# so picking "best" or counting patience on them produces nonsense. Minimum
# guaranteed training length is therefore WARMUP_EPOCHS + RAMP_EPOCHS.
RAMP_END = WARMUP_EPOCHS + RAMP_EPOCHS  # first epoch index where beta==BETA_FINAL

history = []
best_val = float("inf")
best_z = None
patience_counter = 0

for epoch in range(MAX_EPOCHS):
    beta = get_beta(epoch)

    # ---- Train ----
    model.train()
    perm = torch.randperm(len(train_idx))
    train_loss_sum = train_recon_sum = train_kld_sum = 0.0
    n_batches = 0
    for i in range(0, len(perm), BATCH_SIZE):
        bi = perm[i:i + BATCH_SIZE]
        out, mu, lv = model(X_train[bi])
        loss, recon, kld = vae_loss(out, Y_train[bi], mu, lv, beta=beta)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        train_loss_sum  += loss.item()
        train_recon_sum += recon.item()
        train_kld_sum   += kld.item()
        n_batches += 1
    train_loss  = train_loss_sum  / n_batches
    train_recon = train_recon_sum / n_batches
    train_kld   = train_kld_sum   / n_batches

    # ---- Validate ----
    model.eval()
    with torch.no_grad():
        out, mu_v, lv_v = model(X_val)
        v_loss, v_recon, v_kld = vae_loss(out, Y_val, mu_v, lv_v, beta=beta)
        val_loss  = v_loss.item()
        val_recon = v_recon.item()
        val_kld   = v_kld.item()

    current_lr = optimizer.param_groups[0]["lr"]
    history.append({
        "epoch":      epoch + 1,
        "beta":       beta,
        "train_loss": train_loss,
        "train_recon": train_recon,
        "train_kld":  train_kld,
        "val_loss":   val_loss,
        "val_recon":  val_recon,
        "val_kld":    val_kld,
        "lr":         current_lr,
    })

    # ---- Best-model snapshot (post-ramp only) ----
    # During warm-up and ramp, snapshot the latent every epoch so we always
    # have *something* on disk if the job dies, but don't track it as "best"
    # for early-stopping purposes, val_loss isn't comparable while beta
    # is changing.
    post_ramp = epoch >= RAMP_END
    with torch.no_grad():
        mu_full, _ = model.encode(X_full)
        z_now = mu_full.cpu().numpy()
    if not post_ramp:
        # Always overwrite the checkpoint during warm-up/ramp so "latest" is
        # available even if the job times out before reaching full beta.
        np.save(OUT_DIR / "best_z_checkpoint.npy", z_now)
    else:
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            best_z = z_now
            np.save(OUT_DIR / "best_z_checkpoint.npy", best_z)
        else:
            patience_counter += 1

    # LR scheduler only steps on post-ramp val_loss (otherwise it'd "see"
    # the ramp-induced increases and slash lr inappropriately).
    if post_ramp:
        scheduler.step(val_loss)

    if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == RAMP_END:
        phase = "warm" if epoch < WARMUP_EPOCHS else ("ramp" if epoch < RAMP_END else "full")
        print(f"[epoch {epoch + 1:>3d} {phase}] beta={beta:.3f}  "
              f"train={train_loss:>10.0f}  val={val_loss:>10.0f}  "
              f"lr={current_lr:.1e}  best={best_val if post_ramp else float('nan'):>10.0f}  "
              f"patience={patience_counter:>2d}")

    if post_ramp and patience_counter >= PATIENCE:
        print(f"[early stop] no val improvement for {PATIENCE} epochs (post-ramp)")
        break

# If training ended without ever reaching post_ramp (shouldn't happen with
# MAX_EPOCHS=500 and RAMP_END=80, but defensively): fall back to the last
# snapshot.
if best_z is None:
    print("[warn] training ended before ramp completed; using last latent snapshot")
    best_z = z_now
    best_val = val_loss

print(f"[done] best_val={best_val:.2f}, total epochs={len(history)}")


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
pd.DataFrame(history).to_csv(OUT_DIR / "history.tsv", sep="\t", index=False)

run_meta = {
    "trainer": "Plan B",
    "size": SIZE,
    "config": CONFIG,
    "seed": SEED,
    "n_samples": int(n_samples),
    "n_variants": int(n_variants),
    "epochs_trained": int(len(history)),
    "best_val_loss": float(best_val),
    "device": str(device),
    "hyperparameters": {
        "latent_dim": LATENT_DIM,
        "hidden": HIDDEN,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "lr_patience": LR_PATIENCE,
        "lr_factor": LR_FACTOR,
        "lr_min": LR_MIN,
        "beta_final": BETA_FINAL,
        "warmup_epochs": WARMUP_EPOCHS,
        "ramp_epochs": RAMP_EPOCHS,
        "data_split_seed": DATA_SPLIT_SEED,
        "grad_clip_norm": GRAD_CLIP_NORM,
    },
}
with (OUT_DIR / "run.json").open("w") as fh:
    json.dump(run_meta, fh, indent=2)
    fh.write("\n")

latent_df = pd.DataFrame({
    "sample": sample_ids,
    "z1": best_z[:, 0],
    "z2": best_z[:, 1],
})
latent_df.to_csv(OUT_DIR / "latent.tsv", sep="\t", index=False)


# ---------------------------------------------------------------------------
# Sanity checks (collapse detection)
# ---------------------------------------------------------------------------
sanity = []

# (1) Per-dimension standard deviation. A healthy latent has comparable
# spread along both axes; a near-1D collapse shows up as one std much
# smaller than the other.
sd1, sd2 = float(best_z[:, 0].std()), float(best_z[:, 1].std())
sanity.append(f"latent std per dim:    z1={sd1:.4f}  z2={sd2:.4f}")
ratio = max(sd1, sd2) / max(min(sd1, sd2), 1e-12)
sanity.append(f"std ratio (max / min): {ratio:.2f}")
if ratio > 10:
    sanity.append("WARNING: latent dims have very different spreads (>10x). "
                  "Possible near-1D collapse.")

# (2) Singular values of the mean-centred latent. Like a mini-PCA on the
# latent itself: the ratio of singular values is a tighter measure of
# rank than per-dim std (which can mislead if the collapse axis is rotated).
zc = best_z - best_z.mean(axis=0, keepdims=True)
sv = np.linalg.svd(zc, compute_uv=False)
sanity.append(f"singular values:       sv1={sv[0]:.4f}  sv2={sv[1]:.4f}")
sv_ratio = sv[0] / max(sv[1], 1e-12)
sanity.append(f"sv ratio (sv1 / sv2):  {sv_ratio:.2f}")
if sv_ratio > 5:
    sanity.append("WARNING: sv1 / sv2 > 5. Latent is approaching rank-1 "
                  "(snake collapse). Threshold tightened from >20 after the "
                  "Plan A 470k Config 1 run scored 86 here.")

# (3) Latent bounding-box area. Catches the all-axis collapse pattern
# where both dimensions get crushed into a tiny region (the Plan A 470k
# Config 4 case: z1 range 0.2, z2 range 0.18, area 0.04). A healthy 470k
# latent has area in the tens of square units (Config 6 with Plan A had
# ~10 x ~10 = 100). The std-ratio and sv-ratio checks miss this because
# both dimensions collapse proportionally.
z1_range = float(best_z[:, 0].max() - best_z[:, 0].min())
z2_range = float(best_z[:, 1].max() - best_z[:, 1].min())
bbox_area = z1_range * z2_range
sanity.append(f"latent range:          z1_range={z1_range:.4f}  z2_range={z2_range:.4f}")
sanity.append(f"bounding-box area:     {bbox_area:.4f}")
if bbox_area < 1.0:
    sanity.append("WARNING: bounding-box area < 1.0. "
                  "Latent has collapsed to a near-zero region.")
if min(z1_range, z2_range) < 0.5:
    sanity.append("WARNING: smallest latent dimension range < 0.5. "
                  "Latent is effectively 1D.")

# (4) Pairwise distance distribution. If most samples sit on top of each
# other, median pairwise distance will be tiny relative to the max, OR
# the absolute max distance itself will be tiny. Guard the ratio against
# div-by-zero: total collapse (every sample at the same point) gives
# max_d = 0 and would otherwise crash here.
n = len(best_z)
sub = best_z if n <= 200 else best_z[
    np.random.RandomState(0).choice(n, 200, replace=False)
]
diffs = sub[:, None, :] - sub[None, :, :]
dists = np.sqrt((diffs ** 2).sum(axis=-1))
pair_dists = dists[np.triu_indices_from(dists, k=1)]
median_d = float(np.median(pair_dists))
max_d = float(pair_dists.max())
sanity.append(f"pairwise dist:         median={median_d:.4f}  max={max_d:.4f}")
if max_d < 1e-9:
    sanity.append("median / max:          undefined (max=0)")
    sanity.append("WARNING: max pairwise distance is essentially zero. "
                  "Total collapse: all samples at a single point in latent space.")
else:
    md_ratio = median_d / max_d
    sanity.append(f"median / max:          {md_ratio:.4f}")
    if md_ratio < 0.05:
        sanity.append("WARNING: median pairwise dist < 5% of max. "
                      "Most samples have collapsed into a small region.")
    if max_d < 1.0:
        sanity.append("WARNING: max pairwise distance < 1.0. "
                      "Entire population crushed into a tiny region of latent space.")

with (OUT_DIR / "sanity.txt").open("w") as fh:
    fh.write("\n".join(sanity) + "\n")
print("[sanity]")
for line in sanity:
    print(f"  {line}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
# Try to load metadata for colouring by continental region.
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
        print(f"[plot] metadata lacks expected columns; got {list(meta.columns)[:8]}...")
        meta = None
except Exception as e:
    print(f"[plot] could not load metadata ({e}) -- plotting without colouring")
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
ax.set_title(
    f"VAE latent -- {SIZE} config{CONFIG} seed{SEED}\n"
    f"val_loss={best_val:.0f}  epochs={len(history)}  "
    f"sv ratio={sv_ratio:.1f}  bbox area={bbox_area:.2f}"
)
fig.tight_layout()
fig.savefig(OUT_DIR / f"latent_{SIZE}_C{CONFIG}_S{SEED}.png")
plt.close(fig)
print(f"[plot] {OUT_DIR / f'latent_{SIZE}_C{CONFIG}_S{SEED}.png'}")
print(f"[done] outputs in {OUT_DIR}")
