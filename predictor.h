#pragma once
#include "imageIO.h"
#include <vector>
#include <cstdint>

// Existing MED:
int  med_predict(int A, int B, int C);
std::vector<int16_t> compute_residuals_MED_u8(const Image& src);
Image reconstruct_from_residuals_MED(const std::vector<int16_t>& residuals,
                                     const Image& shape);
std::vector<int16_t> compute_residuals_MED_s16(const Image16& src);
Image16 reconstruct_from_residuals_MED_s16(const std::vector<int16_t>& residuals,
                                           const Image16& shape);
Image residuals_visual_rgb8(const std::vector<int16_t>& residuals, const Image& shape);
Image residuals_visual_s16(const std::vector<int16_t>& residuals, const Image16& shape);

// NEW: LS predictors (Gaussian elimination per pixel)
// RGB/Gray (uint8)
std::vector<int16_t> compute_residuals_LS_u8(const Image& src,
                                             int N = 4,
                                             int winW = 4, int winH = 4);
Image reconstruct_from_residuals_LS_u8(const std::vector<int16_t>& residuals,
                                       const Image& shape,
                                       int N = 4,
                                       int winW = 4, int winH = 4);

// RCT int16 (optional LS on RCT)
std::vector<int16_t> compute_residuals_LS_s16(const Image16& src,
                                              int N = 4,
                                              int winW = 4, int winH = 4);
Image16 reconstruct_from_residuals_LS_s16(const std::vector<int16_t>& residuals,
                                          const Image16& shape,
                                          int N = 4,
                                          int winW = 4, int winH = 4);
