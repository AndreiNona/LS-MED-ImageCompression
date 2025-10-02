# Byte2Bit — Predictive, Lossless Image Round Trip

A small C++ project testing **lossless image prediction and reconstruction**.  
It supports **Median Edge Detector (MED)** and **Least-Squares (LS)** prediction, works in both RGB and YUV  domains, and can save/load residuals for later reconstruction.

---

## Features

- **Lossless reconstruction** — reconstructed image pixels match the original exactly
- **Residual visualization** — view prediction residuals as a PNG heatmap
- **Residual file I/O** — save residuals to disk, reconstruct from them later
- **Configurable LS parameters** — number of neighbors, window size
- **Pixel prediction breakdown** — counts how many pixels were predicted by LS vs MED fallback

---

IMG_IN_DIR / IMG_OUT_DIR:  Image source / artefact output directory

IMG_MODE: "rgb" (predict in RGB/Gray), "yuv" , or "ls"

IMG_LS_ON: when IMG_MODE=ls, choose "rgb" or "yuv" for desired color space

IMG_LS_N, IMG_LS_WIN_W, IMG_LS_WIN_H: LS model order and window size.

IMG_SAVE_VIS: save residual visualizations in normal runs

IMG_COMPARE_YUV: enable compare (RGB vs YUV). RGB inputs only

IMG_COMPARE_SAVE_VIS: save residual visuals for both branches in compare mode

IMG_COMPARE_SUFFIX: suffix used to name compare artifacts (e.g., _cmp_rgb_reconstructed.png)

IMG_BATCH_SUMMARY, IMG_COMPARE_SUMMARY: optional for the text summaries

Example of variables for running image processing using YUV:

IMG_COMPARE_SAVE_VIS=fase;
IMG_COMPARE_SUFFIX=_cmp;
IMG_COMPARE_YUV=false;
IMG_IN=Your_path_here;
IMG_LS_N=4;
IMG_LS_ON=YUV;
IMG_LS_WIN_H=4;
IMG_LS_WIN_W=4;
IMG_MODE=ls;
IMG_OUT_DIR=Your_path_here;
IMG_RECURSIVE=false;
IMG_SAVE_RES_VIS=true

---
##Predictors

-MED predictor (fallback): standard median edge detector on neighbors
-LS predictor: Main prediction method, with included lambda constant for less fallback pixels

##Color transform
 -YUV
 For RGB: Y = (R + 2G + B) >> 2, U = B - G, V = R - G
 For Gray: packs Y only into int16.

##Entropy coding (ansResidual)

## Flow

The project works through the subsequent steps

Configuration:
 -Reads environment variables, set up before run or autofilled if empty.
 Image loop:
 -Load image (only 1 or 3 channel supported, alpha is dropped)
 -Predict -> residuals -> ANS compress -> reconstruct -> save outputs
 Statistics:
 -Compute statistics per image (posted to console)
 -Computes batch statistics (saved to file)

