#ifndef BYTE2BITPROJECT1_ANSRESIDUAL_H
#define BYTE2BITPROJECT1_ANSRESIDUAL_H

// (kept only so existing includes don't break)
class ansResidual { };

#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ans {

    // ---- Public constants ----
    static constexpr uint32_t RANS_L     = 1u << 12;   // normalization (L=4096)
    static constexpr uint32_t MAX_SYM    = 4095;       // 0..4095 symbols (12-bit)
    static constexpr uint16_t ESC_SYM    = MAX_SYM;    // escape code
    static constexpr uint32_t ALPHABET   = MAX_SYM + 1;
    static constexpr uint32_t FILE_MAGIC = 0x534E4152; // 'RANS' (LE)


    struct Encoded {
        size_t escapes   = 0;   // number of escape residuals (|value| >= MAX_SYM/zigzag)
        size_t n_syms    = 0;   // number of symbols encoded
        size_t ans_bytes = 0;   // size of the ANS payload in bytes (container section only)
    };

    Encoded compress_to_file(const std::vector<int16_t>& residuals,
                             int mode, int w, int h, int c,
                             const std::string& outPath);

    std::vector<int16_t> decompress_file(const std::string& inPath);

} // namespace ans
#endif // BYTE2BITPROJECT1_ANSRESIDUAL_H
