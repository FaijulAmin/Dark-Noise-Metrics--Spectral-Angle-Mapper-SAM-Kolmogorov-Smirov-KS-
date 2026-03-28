## @file    spectral_metrics.py
## @brief   SAM + KS distance evaluation — dark frame vs model output
## @dataset configurable — calibrated to Indian Pines (145 x 145 x 220, uint16)
##
## @metric  SAM  Spectral Angle Mapper        Kruse et al. 1993
## @metric  KS   Kolmogorov-Smirnov Distance  per-pixel, per-band
##
## @usage
##   python spectral_metrics.py
##   python spectral_metrics.py --gen-model
##   python spectral_metrics.py --pixel 70 80
##
## @inputs
##   data/Indian_pines.mat   hyperspectral cube  (H, W, B)
##   data/model_output.npy   model output        (H, W, B)
##   data/dark_frame.npy     generated once, reused on subsequent runs
##
## @outputs
##   results/sam_map.png     SAM spatial map
##   results/ks_map.png      KS spatial map
##   results/ks_bands.png    per-band KS curve
##   results/sam_map.npy     per-pixel SAM     (H, W)
##   results/ks_map.npy      per-pixel KS D    (H, W)
##   results/ks_bands.npy    per-band  KS D    (B,)

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy             as np
import scipy.io          as sio
from scipy.stats import ks_2samp


# ── plot theme ────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor" : "#060606",
    "axes.facecolor"   : "#0a0a0a",
    "axes.edgecolor"   : "#1e1e1e",
    "axes.labelcolor"  : "#555",
    "axes.titlecolor"  : "#666",
    "axes.grid"        : True,
    "grid.color"       : "#141414",
    "grid.linewidth"   : 0.6,
    "xtick.color"      : "#333",
    "ytick.color"      : "#333",
    "text.color"       : "#888",
    "font.family"      : "monospace",
    "font.size"        : 8,
    "lines.linewidth"  : 1.2,
})

GOLD = "#c8a96e"
BLUE = "#4a9eff"


# ── dataset config ────────────────────────────────────────────────────────────

CONFIG = {
    "mat_key"   : None,  # None = auto-detect first non-private key
    "norm_max"  : True,  # normalise cube to [0, 1] on load
    "dark_seed" : 42,
    "model_snr" : 15,
}


# ── paths ─────────────────────────────────────────────────────────────────────

MAT_PATH   = "data/Indian_pines.mat"
MODEL_PATH = "data/model_output.npy"   ##< swap to real model output here
DARK_PATH  = "data/dark_frame.npy"
OUT_DIR    = "results"

os.makedirs("data",    exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)


# ── data loaders ──────────────────────────────────────────────────────────────

def load_cube(path, mat_key=None, normalise=True):
    ## @brief  Load a hyperspectral cube from .mat or .npy.
    ## @param  path       path to input file (.mat or .npy)
    ## @param  mat_key    key to extract from .mat — None = auto-detect
    ## @param  normalise  if True, scale to [0, 1]
    ## @return float32 array  (H, W, B)
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".npy":
        cube = np.load(path).astype(np.float32)
    elif ext == ".mat":
        mat = sio.loadmat(path)
        if mat_key is None:
            mat_key = [k for k in mat if not k.startswith("_")][0]
        cube = mat[mat_key].astype(np.float32)
    else:
        raise ValueError(f"unsupported format: {ext}  (expected .mat or .npy)")

    if normalise:
        cube /= cube.max()
    return cube


def make_dark_frame(shape, seed=42):
    ## @brief  Synthetic dark frame: thermal noise + column readout + hot pixels.
    ## @param  shape  (H, W, B) matching the hyperspectral cube
    ## @param  seed   RNG seed for reproducibility
    ## @return float32 array  (H, W, B), clipped >= 0
    rng     = np.random.default_rng(seed)
    H, W, B = shape

    base    = 0.02 + 0.008 * np.sin(np.linspace(0, 3, B))
    thermal = rng.normal(0, 0.004, (H, W, B))
    readout = rng.normal(0, 0.002, (H, 1, B)) * np.ones((1, W, 1))
    hot     = (rng.random((H, W, B)) > 0.998) * rng.uniform(0.1, 0.3, (H, W, B))

    dark = base[None, None, :] + thermal + readout + hot
    return np.clip(dark, 0, None).astype(np.float32)


def make_synthetic_model(dark, snr=15, seed=7):
    ## @brief  Synthetic model output: gaussian noise added to dark frame.
    ## @note   Replace with real model_output.npy for actual evaluation.
    ## @param  dark   dark frame  (H, W, B)
    ## @param  snr    signal-to-noise ratio
    ## @param  seed   RNG seed
    ## @return float32 array  (H, W, B), clipped >= 0
    rng   = np.random.default_rng(seed)
    noise = rng.normal(0, dark / (snr + 1e-6), dark.shape).astype(np.float32)
    return np.clip(dark + noise, 0, None).astype(np.float32)


# ── metrics ───────────────────────────────────────────────────────────────────

def sam_map(ref, tgt):
    ## @brief  Per-pixel Spectral Angle Mapper.
    ## @detail theta = arccos( dot(a,b) / (|a| |b|) )
    ##         theta = 0 -> identical,  theta = pi/2 -> orthogonal
    ## @param  ref  reference array  (H, W, B)
    ## @param  tgt  target array     (H, W, B)
    ## @return float32 angle map     (H, W)  in radians
    dot   = np.sum(ref * tgt, axis=-1)
    normR = np.linalg.norm(ref, axis=-1)
    normT = np.linalg.norm(tgt, axis=-1)
    denom = normR * normT
    cos   = np.where(denom > 0, dot / np.clip(denom, 1e-10, None), 0)
    return np.arccos(np.clip(cos, -1, 1)).astype(np.float32)


def ks_map(ref, tgt):
    ## @brief  Per-pixel KS distance, treating each pixel's spectrum as a 1-D sample.
    ## @detail D = 0 -> identical distributions,  D = 1 -> no overlap
    ## @param  ref  reference array  (H, W, B)
    ## @param  tgt  target array     (H, W, B)
    ## @return float32 KS D map      (H, W)
    H, W, B = ref.shape
    out   = np.zeros((H, W), dtype=np.float32)
    ref_f = ref.reshape(-1, B)
    tgt_f = tgt.reshape(-1, B)
    for i in range(H * W):
        d, _ = ks_2samp(ref_f[i], tgt_f[i])
        out.flat[i] = d
    return out


def ks_per_band(ref, tgt):
    ## @brief  KS distance per spectral band, pooled over all spatial pixels.
    ## @param  ref  reference array  (H, W, B)
    ## @param  tgt  target array     (H, W, B)
    ## @return float32 array         (B,)
    H, W, B = ref.shape
    ref_f   = ref.reshape(-1, B)
    tgt_f   = tgt.reshape(-1, B)
    return np.array([ks_2samp(ref_f[:, b], tgt_f[:, b])[0] for b in range(B)])


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_sam_map(sam, out):
    ## @brief  SAM spatial heatmap.
    ## @param  sam  angle map in radians  (H, W)
    ## @param  out  output filepath (.png)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.degrees(sam), cmap="inferno", vmin=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("theta (deg)", color="#444", fontsize=7)
    cb.outline.set_edgecolor("#1e1e1e")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#333", fontsize=7)
    ax.set_title("SAM  [degrees]", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  saved -> {out}")


def plot_ks_map(ks_d, out):
    ## @brief  KS distance spatial heatmap.
    ## @param  ks_d  KS D map  (H, W)
    ## @param  out   output filepath (.png)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(ks_d, cmap="magma", vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("D", color="#444", fontsize=7)
    cb.outline.set_edgecolor("#1e1e1e")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#333", fontsize=7)
    ax.set_title("KS DISTANCE  [per pixel]", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  saved -> {out}")


def plot_ks_bands(ks_bands, out):
    ## @brief  KS distance per spectral band curve.
    ## @param  ks_bands  KS D per band  (B,)
    ## @param  out       output filepath (.png)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks_bands, color=GOLD, lw=1.2)
    ax.fill_between(range(len(ks_bands)), ks_bands, alpha=0.15, color=GOLD)
    ax.set_xlabel("band index")
    ax.set_ylabel("KS D")
    ax.set_title("KS DISTANCE PER SPECTRAL BAND", fontsize=7)
    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  saved -> {out}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ## @brief  CLI entry point. Orchestrates load -> generate -> compute -> plot.
    ## @note   Use --gen-model to force regenerate synthetic model_output.npy.
    ##         Use --pixel ROW COL for a single-pixel deep-dive plot.

    parser = argparse.ArgumentParser(description="SAM + KS spectral metrics")
    parser.add_argument("--gen-model", action="store_true",
                        help="force regenerate synthetic model_output.npy")
    args = parser.parse_args()

    # -- load cube ------------------------------------------------------------
    print("loading cube ...")
    cube    = load_cube(MAT_PATH, mat_key=CONFIG["mat_key"],
                        normalise=CONFIG["norm_max"])
    H, W, B = cube.shape
    print(f"  cube   {cube.shape}  dtype {cube.dtype}")

    # -- dark frame: load or generate -----------------------------------------
    if os.path.exists(DARK_PATH):
        dark = np.load(DARK_PATH)
        print(f"  dark   loaded from {DARK_PATH}")
    else:
        print("  generating dark frame ...")
        dark = make_dark_frame(cube.shape, seed=CONFIG["dark_seed"])
        np.save(DARK_PATH, dark)
        print(f"  dark   saved -> {DARK_PATH}")

    # -- model output: load or generate ---------------------------------------
    if os.path.exists(MODEL_PATH) and not args.gen_model:
        model = np.load(MODEL_PATH).astype(np.float32)
        print(f"  model  loaded from {MODEL_PATH}")
    else:
        print("  generating synthetic model output ...")
        model = make_synthetic_model(dark, snr=CONFIG["model_snr"])
        np.save(MODEL_PATH, model)
        print(f"  model  saved -> {MODEL_PATH}")

    assert dark.shape == model.shape == cube.shape, (
        f"shape mismatch: dark {dark.shape}  model {model.shape}  cube {cube.shape}"
    )

    # -- compute metrics ------------------------------------------------------
    print("computing SAM ...")
    sam  = sam_map(dark, model)

    print("computing KS per pixel ... (~30s for 145x145)")
    ks_d = ks_map(dark, model)

    print("computing KS per band ...")
    ks_b = ks_per_band(dark, model)

    # -- summary printout -----------------------------------------------------
    sam_deg = np.degrees(sam)
    print()
    print("-" * 50)
    print(f"  SAM  mean   {sam_deg.mean():.4f} deg")
    print(f"  SAM  median {np.median(sam_deg):.4f} deg")
    print(f"  SAM  max    {sam_deg.max():.4f} deg")
    print(f"  KS   mean   {ks_d.mean():.4f}")
    print(f"  KS   median {np.median(ks_d):.4f}")
    print(f"  KS   max    {ks_d.max():.4f}")
    print("-" * 50)

    # -- save metric arrays ---------------------------------------------------
    np.save(f"{OUT_DIR}/sam_map.npy",  sam)
    np.save(f"{OUT_DIR}/ks_map.npy",   ks_d)
    np.save(f"{OUT_DIR}/ks_bands.npy", ks_b)
    print(f"  metrics saved -> {OUT_DIR}/")

    # -- plots ----------------------------------------------------------------
    print("plotting ...")
    plot_sam_map(sam,  out=f"{OUT_DIR}/sam_map.png")
    plot_ks_map(ks_d,  out=f"{OUT_DIR}/ks_map.png")
    plot_ks_bands(ks_b, out=f"{OUT_DIR}/ks_bands.png")

    print("done.")


if __name__ == "__main__":
    main()