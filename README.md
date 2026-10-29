# Multi-Scale Detail Enhancement with Laplacian Pyramids

This project implements a multi-scale image enhancement pipeline using Gaussian and Laplacian pyramids. An image is split into frequency bands, each band is amplified with its own weight, and the image is rebuilt, which sharpens fine textures and boosts local contrast without a global filter.

## Project Overview

A single sharpening filter treats every detail the same way. A Laplacian pyramid separates an image into layers of decreasing detail (fine texture, mid-level structure, coarse shapes), so each scale can be strengthened or left alone independently. This repository contains a from-scratch implementation built on OpenCV and NumPy.

## Methodology

The pipeline has four steps:

1. **Pre-processing:** Every input is resized so its shortest edge is exactly 512 px. The aspect ratio is preserved and `cv2.INTER_LANCZOS4` interpolation is used.
2. **Gaussian pyramid:** 5 levels, built with repeated `cv2.pyrDown`. The image is converted to `float32` so no precision is lost during subtraction.
3. **Laplacian pyramid:** Each level is the difference between a Gaussian level and the upsampled next level. `cv2.pyrUp` is called with an explicit `dstsize` so odd-sized images do not cause off-by-one shape mismatches. The last level (L4) is the low-pass residual.
4. **Selective amplification and reconstruction:** Each layer is multiplied by its weight, then the image is rebuilt from coarse to fine by upsampling and adding. The result is clipped to `[0, 255]` and converted to `uint8`.

The default weights, from the finest layer (L0) to the low-pass residual (L4):

```python
beta_weights = [1.0, 1.05, 1.15, 1.20, 1.0]
```

L0 is left unchanged on purpose, because it holds the finest detail and most of the pixel noise.

## Experimental Findings

* **Fine detail:** Raising the weights of L1 and L2 made small details such as hair and fabric much sharper, but it also amplified JPEG compression noise in smooth areas.
* **Local contrast:** Raising the L3 weight (for example to 1.35) improved depth and shadows, and worked best on nature and portrait images without adding visible pixel noise.
* **Limit test:** When the weights went above roughly 1.25 on several layers at once, the RGB channels hit the 255 limit. Clipping broke the ratio between channels and caused a yellow/sepia color shift.
* **Possible improvement:** Converting the image to YCrCb and amplifying only the luma (Y) channel should avoid the color shift.

## Repository Structure

```text
laplacian_pyramid_detail_enhancement/
├── images/          # create this folder and put your input images here
├── main.py
├── requirements.txt
└── README.md
```

The `outputs/` folder is created automatically when the script runs.

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/usakgun/laplacian_pyramid_detail_enhancement.git
   cd laplacian_pyramid_detail_enhancement
   ```
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Create an `images/` folder in the project root and add your own images to it. Sample images are not included in this repository.
4. Run the script:
   ```bash
   python main.py
   ```

Every image in `images/` is processed, and the result is saved to `outputs/<name>_enhanced.<ext>`.

> **Note:** Only files ending in lowercase `.jpg`, `.jpeg` or `.png` are picked up. On Linux and macOS, a file such as `photo.JPG` is skipped, so rename it first.

## Technologies

* **Python**
* **OpenCV:** Resizing, pyramid down/up-sampling, arithmetic on image layers.
* **NumPy:** Array operations and clipping.
