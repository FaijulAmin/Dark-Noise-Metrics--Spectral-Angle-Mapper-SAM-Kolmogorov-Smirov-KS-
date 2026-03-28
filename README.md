# Dark-Noise-Metrics--Spectral-Angle-Mapper-SAM-Kolmogorov-Smirov-KS-

Hyperspectral evaluation: Spectral Angle Mapper (SAM) and Kolmogorov–Smirnov (KS) metrics comparing a dark-frame reference to model output. See `spectral_metrics.py`.

## Purpose and data

The program compares a dark frame reference to a target volume on the Indian Pines hyperspectral cube. Data come from a .mat file, are scaled to a common range, and must match the dark and model arrays in size.

## Method

“spectral_metrics.py” loads or generates a synthetic dark frame. That frame uses Gaussian noise for thermal variation and for column readout, plus a non-Gaussian hot-pixel component (sparse random spikes). The placeholder model output adds another layer of Gaussian noise whose scale depends on the dark frame and a chosen SNR. The script then computes SAM per pixel and KS per pixel and per band, and writes numeric arrays and figures to the results.

## Outputs and use

The run produces SAM and KS maps, a per band KS curve, and summary numbers in the console. SAM summarizes spectral similarity between reference and target; KS highlights spatial and wavelength differences. For evaluation of a real algorithm, the synthetic model needs to be replaced with actual model outputs; the Gaussian noise model is mainly a stand-in for testing the metric pipeline.
