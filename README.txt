# Byte2Bit — Predictive, Lossless Image Round Trip

A small C++ project testing **lossless image prediction and reconstruction**.  
It supports **Median Edge Detector (MED)** and **Least-Squares (LS)** prediction, works in both RGB and RCT  domains, and can save/load residuals for later reconstruction.

---

## Features

- **Modes**: MED on RGB, MED on RCT, LS on RGB, LS on RCT
- **Lossless reconstruction** — reconstructed image pixels match the original exactly
- **Residual visualization** — view prediction residuals as a PNG heatmap
- **Residual file I/O** — save residuals to disk, reconstruct from them later
- **Configurable LS parameters** — number of neighbors, window size
- **Pixel prediction breakdown** — counts how many pixels were predicted by LS vs MED fallback

---

# MED on RGB, save residuals
 img.png --mode=rgb --save-res=rgb_res.r16

# Reconstruct from residual file
 --load-res=rgb_res.r16

# MED on RCT, visualize residuals
 img.png --mode=rct --save-res-vis

# LS on RGB, N=4, win=4x4, visualize
 img.png --mode=ls --ls-on=rgb --ls-N=4 --ls-win=4x4 --save-res-vis

# LS on RCT, save residuals
 img.png --mode=ls --ls-on=rct --save-res=ls_rct.r16

 Average prediction based on testing is 98% LS
 (obviously diverges based on source image, but since it is an average should give an idea of how 'aggressive' we are when switching to MED)