#include "ansResidual.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ans {

// ------------------ residual / symbol mapping (internal) ------------------
static inline uint32_t zigzag16(int16_t r) {
    // interleave
    uint32_t u = static_cast<uint32_t>(r);
    return (u << 1) ^ static_cast<uint32_t>(r >> 15);
}
static inline int16_t unzigzag16(uint32_t z) {
    return static_cast<int16_t>((z >> 1) ^ (~(z & 1) + 1));
}

struct Symbolized {
    std::vector<uint16_t> syms; // values in [0..MAX_SYM], ESC_SYM marks escape
    std::vector<int16_t>  esc;  // raw outliers, in order
    size_t escapes = 0;
};

static Symbolized symbolize_residuals(const std::vector<int16_t>& residuals) {
    Symbolized S;
    S.syms.resize(residuals.size());
    for (size_t i = 0; i < residuals.size(); ++i) {
        uint32_t z = zigzag16(residuals[i]);
        if (z < MAX_SYM) {
            S.syms[i] = static_cast<uint16_t>(z);
        } else {
            S.syms[i] = ESC_SYM;
            S.esc.push_back(residuals[i]);
            ++S.escapes;
        }
    }
    return S;
}

static std::vector<int16_t> unsymbolize_residuals(const std::vector<uint16_t>& syms,
                                                  const std::vector<int16_t>& esc)
{
    std::vector<int16_t> out;
    out.resize(syms.size());
    size_t ei = 0;
    for (size_t i = 0; i < syms.size(); ++i) {
        uint16_t s = syms[i];
        out[i] = (s == ESC_SYM) ? esc[ei++] : unzigzag16(s);
    }
    return out;
}

// ----------------------------- static model -------------------------------
struct Model {
    uint32_t L = RANS_L;
    std::vector<uint16_t> freq;     // size = ALPHABET
    std::vector<uint32_t> cdf;      // size = ALPHABET
    std::vector<uint16_t> lut_sym;  // size = L
};

static Model build_model(const std::vector<uint16_t>& syms) {
    Model m;
    m.freq.assign(ALPHABET, 0);
    for (auto s : syms) m.freq[s]++;

    uint64_t total = 0;
    for (auto f : m.freq) total += f;
    if (total == 0) { m.freq[0] = static_cast<uint16_t>(m.L); total = m.L; }

    std::vector<double> prob(ALPHABET, 0.0);
    for (size_t s = 0; s < ALPHABET; ++s)
        prob[s] = static_cast<double>(m.freq[s]) / static_cast<double>(std::max<uint64_t>(1, total));

    uint32_t sum = 0;
    for (size_t s = 0; s < ALPHABET; ++s) {
        m.freq[s] = static_cast<uint16_t>(std::max(1, static_cast<int>(std::lround(prob[s] * m.L))));
        sum += m.freq[s];
    }
    while (sum > m.L) {
        size_t idx = static_cast<size_t>(std::distance(m.freq.begin(),
                       std::max_element(m.freq.begin(), m.freq.end())));
        if (m.freq[idx] > 1) { m.freq[idx]--; sum--; } else break;
    }
    while (sum < m.L) {
        size_t idx = static_cast<size_t>(std::distance(m.freq.begin(),
                       std::min_element(m.freq.begin(), m.freq.end())));
        m.freq[idx]++; sum++;
    }

    m.cdf.resize(ALPHABET);
    uint32_t c = 0;
    for (size_t s = 0; s < ALPHABET; ++s) { m.cdf[s] = c; c += m.freq[s]; }

    m.lut_sym.resize(m.L);
    for (uint32_t s = 0; s < ALPHABET; ++s) {
        uint32_t f = m.freq[s], start = m.cdf[s];
        for (uint32_t i = 0; i < f; ++i) m.lut_sym[start + i] = static_cast<uint16_t>(s);
    }
    return m;
}

// ------------------------------- rANS32 -----------------------------------
namespace rans32 {
    static constexpr int PREC = 12;            // log2(L)
    static constexpr uint32_t L  = RANS_L;     // 4096

    static std::vector<uint8_t> encode(const std::vector<uint16_t>& syms,
                                       const Model& m)
    {
        std::vector<uint8_t> out;
        out.reserve(syms.size() / 2 + 16);

        uint32_t x = 1u << 16;
        auto put  = [&](uint8_t b) { out.push_back(b); };

        for (size_t i = syms.size(); i-- > 0;) {
            uint16_t s  = syms[i];
            uint32_t f  = m.freq[s];
            uint32_t cf = m.cdf[s];

            while (x >= (f << (32 - PREC))) {
                put(static_cast<uint8_t>(x & 0xFF));
                x >>= 8;
            }
            x = (x / f) * L + (x % f) + cf;
        }
        // flush 4 bytes of state
        put(static_cast<uint8_t>(x & 0xFF)); x >>= 8;
        put(static_cast<uint8_t>(x & 0xFF)); x >>= 8;
        put(static_cast<uint8_t>(x & 0xFF)); x >>= 8;
        put(static_cast<uint8_t>(x & 0xFF));
        return out;
    }

    static std::vector<uint16_t> decode(const std::vector<uint8_t>& in,
                                        size_t n_syms,
                                        const Model& m)
    {
        std::vector<uint16_t> out(n_syms);
        size_t ip = in.size();

        auto get = [&]() -> uint32_t {
            if (ip == 0) throw std::runtime_error("rANS underflow");
            return static_cast<uint32_t>(in[--ip]);
        };

        uint32_t x = 0;
        x |= get();
        x |= get() << 8;
        x |= get() << 16;
        x |= get() << 24;

        for (size_t i = 0; i < n_syms; ++i) {
            uint32_t slot = x & (L - 1);
            uint16_t s    = m.lut_sym[slot];
            out[i]        = s;

            uint32_t cf = m.cdf[s], f = m.freq[s];
            x = f * (x >> PREC) + (slot - cf);

            while (x < (1u << 16)) x = (x << 8) | get();
        }
        return out;
    }
} // namespace rans32

// --------------------------- container I/O --------------------------------
struct Packed {
    int mode = 0;              // 0 = u8/RGB residuals; 1 = s16 domain residuals
    int w = 0, h = 0, c = 0;
    uint64_t n_syms = 0;
    Model model;
    std::vector<uint8_t>  ans_bytes;
    std::vector<int16_t>  escapes;
};

static void save_file(const std::string& path, const Packed& P) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("open write: " + path);

    uint32_t magic = FILE_MAGIC;
    uint32_t L     = P.model.L;
    uint32_t ALPH  = static_cast<uint32_t>(P.model.freq.size());
    uint64_t esc_count = static_cast<uint64_t>(P.escapes.size());
    uint64_t esc_bytes = esc_count * 2;
    uint64_t ans_size  = static_cast<uint64_t>(P.ans_bytes.size());

    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&P.mode), 4);
    f.write(reinterpret_cast<const char*>(&P.w), 4);
    f.write(reinterpret_cast<const char*>(&P.h), 4);
    f.write(reinterpret_cast<const char*>(&P.c), 4);
    f.write(reinterpret_cast<const char*>(&P.n_syms), 8);
    f.write(reinterpret_cast<const char*>(&L), 4);
    f.write(reinterpret_cast<const char*>(&ALPH), 4);
    f.write(reinterpret_cast<const char*>(P.model.freq.data()),
            static_cast<std::streamsize>(sizeof(uint16_t) * ALPH));
    f.write(reinterpret_cast<const char*>(&esc_count), 8);
    f.write(reinterpret_cast<const char*>(&esc_bytes), 8);
    f.write(reinterpret_cast<const char*>(&ans_size), 8);
    if (esc_bytes)
        f.write(reinterpret_cast<const char*>(P.escapes.data()),
                static_cast<std::streamsize>(esc_bytes));
    if (ans_size)
        f.write(reinterpret_cast<const char*>(P.ans_bytes.data()),
                static_cast<std::streamsize>(ans_size));

    if (!f) throw std::runtime_error("write failed: " + path);
}

static Packed load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("open read: " + path);

    Packed P{};
    uint32_t magic = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != FILE_MAGIC) throw std::runtime_error("bad magic");

    f.read(reinterpret_cast<char*>(&P.mode), 4);
    f.read(reinterpret_cast<char*>(&P.w), 4);
    f.read(reinterpret_cast<char*>(&P.h), 4);
    f.read(reinterpret_cast<char*>(&P.c), 4);
    f.read(reinterpret_cast<char*>(&P.n_syms), 8);

    uint32_t L = 0, ALPH = 0;
    f.read(reinterpret_cast<char*>(&L), 4);
    f.read(reinterpret_cast<char*>(&ALPH), 4);
    P.model.L = L;
    P.model.freq.resize(ALPH);
    f.read(reinterpret_cast<char*>(P.model.freq.data()),
           static_cast<std::streamsize>(sizeof(uint16_t) * ALPH));

    // rebuild CDF + LUT
    P.model.cdf.resize(ALPH);
    uint32_t cdf = 0;
    for (uint32_t s = 0; s < ALPH; ++s) { P.model.cdf[s] = cdf; cdf += P.model.freq[s]; }
    P.model.lut_sym.resize(L);
    for (uint32_t s = 0; s < ALPH; ++s) {
        uint32_t fsz = P.model.freq[s], start = P.model.cdf[s];
        for (uint32_t i = 0; i < fsz; ++i) P.model.lut_sym[start + i] = static_cast<uint16_t>(s);
    }

    uint64_t esc_count = 0, esc_bytes = 0, ans_size = 0;
    f.read(reinterpret_cast<char*>(&esc_count), 8);
    f.read(reinterpret_cast<char*>(&esc_bytes), 8);
    f.read(reinterpret_cast<char*>(&ans_size), 8);

    P.escapes.resize(static_cast<size_t>(esc_count));
    if (esc_bytes)
        f.read(reinterpret_cast<char*>(P.escapes.data()),
               static_cast<std::streamsize>(esc_bytes));

    P.ans_bytes.resize(static_cast<size_t>(ans_size));
    if (ans_size)
        f.read(reinterpret_cast<char*>(P.ans_bytes.data()),
               static_cast<std::streamsize>(ans_size));

    if (!f) throw std::runtime_error("read failed: " + path);
    return P;
}

// ---------------------------- public API ----------------------------------
Encoded compress_to_file(const std::vector<int16_t>& residuals,
                         int mode, int w, int h, int c,
                         const std::string& outPath)
{
    Encoded info{};

    Symbolized S = symbolize_residuals(residuals);
    Model M      = build_model(S.syms);
    std::vector<uint8_t> ans_bytes = rans32::encode(S.syms, M);

    Packed P;
    P.mode     = mode;
    P.w        = w;
    P.h        = h;
    P.c        = c;
    P.n_syms   = static_cast<uint64_t>(S.syms.size());
    P.model    = std::move(M);
    P.ans_bytes= std::move(ans_bytes);
    P.escapes  = std::move(S.esc);

    save_file(outPath, P);

    info.escapes   = P.escapes.size();
    info.n_syms    = static_cast<size_t>(P.n_syms);
    info.ans_bytes = P.ans_bytes.size();
    return info;
}

std::vector<int16_t> decompress_file(const std::string& inPath) {
    Packed P = load_file(inPath);
    std::vector<uint16_t> syms = rans32::decode(P.ans_bytes, static_cast<size_t>(P.n_syms), P.model);
    return unsymbolize_residuals(syms, P.escapes);
}

} // namespace ans
