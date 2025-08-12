#pragma once
#include "imageIO.h"
#include <vector>
#include <cstdint>
#include <string>

// mode: 0 = RGB(u8); 1 = RCT(s16)
void save_residuals(const std::string& path,
                    int mode, int w, int h, int c,
                    const std::vector<int16_t>& residuals);

struct ResidualFile {
    int mode;
    int w, h, c;
    std::vector<int16_t> residuals;
};

ResidualFile load_residuals(const std::string& path);
