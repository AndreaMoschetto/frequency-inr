# frequency-inr

Pipeline for Implicit Neural Representations (INRs) on image spectra using frequency domain analysis (FFT & DCT).

## Overview

This project benchmarks the reconstruction quality of images using Coordinate-Based Neural Networks (SIREN). Unlike standard approaches that map coordinates $(x,y)$ directly to pixel colors $(R,G,B)$, this framework explores learning the image representation in the **Frequency Domain**.

We compare three different modalities:
1.  **Spatial Domain (Baseline):** Fitting the network directly on pixel values.
2.  **Discrete Cosine Transform (DCT):** Fitting the network on the DCT-II coefficients.
3.  **Fourier Domain (FFT):** Fitting the network on the complex spectrum (Real + Imaginary parts).

## Usage

You can reproduce the study using the commands below. The pipeline consists of two steps for each method:
1.  **Encode:** Trains the SIREN network to learn the image representation.
2.  **Decode:** Reconstructs the image from the trained weights.

### Prerequisites

Ensure you have `pixi` installed to manage dependencies and run the scripts.

### 1. Spatial Domain (Default)
The baseline approach. The network learns the mapping from coordinates to pixels directly.

```bash
# Train the model (1000 iterations)
pixi run python -m bin.encode lena.png ./outputs/spatial.bin --config default --mode full

# Reconstruct the image
pixi run python -m bin.decode ./outputs/spatial.bin 512x512 ./outputs/rec_spatial.png --config default --mode full

```

### 2. Discrete Cosine Transform (DCT)

The network learns the DCT coefficients. This method typically offers superior energy compaction and texture preservation.

```bash
# Train the model on DCT spectrum
pixi run python -m bin.encode lena.png ./outputs/dct.bin --config dct --mode full

# Reconstruct using Hybrid DC Transplant (requires original image)
pixi run python -m bin.decode ./outputs/dct.bin 512x512 ./outputs/rec_dct.png --config dct --mode full --original_image lena.png

```

### 3. Fourier Domain (FFT)

The network learns the Real and Imaginary parts of the 2D FFT spectrum.

```bash
# Train the model on FFT spectrum
pixi run python -m bin.encode lena.png ./outputs/fft.bin --config fourier --mode full

# Reconstruct using Hybrid DC Transplant (requires original image)
pixi run python -m bin.decode ./outputs/fft.bin 512x512 ./outputs/rec_fft.png --config fourier --mode full --original_image lena.png

```

---

## Credits

This project is based on the framework provided in the [ICIP 2024 INR Images Tutorial](https://github.com/aegroto/icip-2024-inr-images-tutorial) by **aegroto**.
The original pipeline for spatial image compression has been extended in this repository to support frequency domain analysis (FFT/DCT) and Implicit Representation Learning on image spectra.