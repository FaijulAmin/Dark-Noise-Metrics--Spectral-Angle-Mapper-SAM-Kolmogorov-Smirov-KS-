# inputs_outputs_notes.py
# Reference only: no logic here. Describes spectral_metrics.py inputs, outputs, and flow.
# -----------------------------------------------------------------------------

# WHAT GETS COMPARED
# The metrics (SAM and KS) always compare two arrays of the same shape:
#   reference  = dark frame   (data/dark_frame.npy, or generated on first run)
#   target     = model output (data/model_output.npy, or generated synthetic)
# The Indian Pines cube is loaded so dark/model match its (H, W, B). The cube
# pixels are not used inside SAM or KS; only shape and dtype pipeline matter.

# -----------------------------------------------------------------------------
# INPUTS (files the script reads or expects)
#
# data/Indian_pines.mat
#   Hyperspectral cube loaded by scipy.io.loadmat. Variable key: first non-
#   underscore key in the file if CONFIG["mat_key"] is None. Shape (H, W, B).
#   Optionally max-normalized to [0, 1] when CONFIG["norm_max"] is True.
#
# data/dark_frame.npy  (optional on first run)
#   If missing: created with make_dark_frame(cube.shape), saved here, reused
#   later. Shape (H, W, B), float32. Synthetic: baseline + Gaussian thermal +
#   Gaussian column readout + sparse hot pixels.
#
# data/model_output.npy  (optional)
#   If present and you do not pass --gen-model: loaded as the model volume.
#   If missing or --gen-model: overwritten by make_synthetic_model (dark +
#   Gaussian noise scaled by CONFIG["model_snr"]). Shape must match cube.

# -----------------------------------------------------------------------------
# OUTPUTS (written under results/ by default; OUT_DIR in spectral_metrics.py)
#
# results/sam_map.npy
#   Per-pixel Spectral Angle Mapper between dark and model. Shape (H, W).
#   Values in radians (plots convert to degrees).
#
# results/ks_map.npy
#   Per-pixel two-sample Kolmogorov-Smirnov D between the two spectra at each
#   pixel (each spectrum is length B along bands). Shape (H, W). D in [0, 1].
#
# results/ks_bands.npy
#   One KS D per spectral band: all spatial pixels pooled for that band for
#   dark vs model. Shape (B,).
#
# results/sam_map.png
#   Heatmap of SAM in degrees.
#
# results/ks_map.png
#   Heatmap of per-pixel KS D.
#
# results/ks_bands.png
#   Curve of KS D vs band index.

# -----------------------------------------------------------------------------
# CONSOLE
# Prints cube shape, whether dark/model were loaded or generated, then mean /
# median / max for SAM (degrees) and for KS D after computation.

# -----------------------------------------------------------------------------
# COMMAND LINE
#   python spectral_metrics.py
#       Normal run: use existing dark_frame.npy and model_output.npy if present.
#   python spectral_metrics.py --gen-model
#       Regenerates synthetic data/model_output.npy even if it already exists.

# -----------------------------------------------------------------------------
# CONFIG (edit CONFIG in spectral_metrics.py)
#   mat_key     None = auto-detect .mat variable; else string key name.
#   norm_max    If True, divide cube by its max on load.
#   dark_seed   RNG seed for dark frame generation.
#   model_snr   SNR used only when building synthetic model output.
