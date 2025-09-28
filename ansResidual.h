
#ifndef BYTE2BITPROJECT1_ANSRESIDUAL_H
#define BYTE2BITPROJECT1_ANSRESIDUAL_H


class ansResidual {
};

#pragma once
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>

namespace ans {

// ------------------------- Config -------------------------
static constexpr uint32_t RANS_L       = 1u << 12;   // normalization (L=4096)
static constexpr uint32_t MAX_SYM      = 4095;       // 0..4095 symbols (12-bit)
static constexpr uint16_t ESC_SYM      = MAX_SYM;    // escape code
static constexpr uint32_t ALPHABET     = MAX_SYM + 1; // includes ESC
static constexpr uint32_t FILE_MAGIC   = 0x534E4152; // 'RANS' (LE)

// ------------------ Residual / Symbol mapping -------------
inline uint32_t zigzag16(int16_t r) {
    uint32_t u = (uint32_t)r;
    return (u << 1) ^ (uint32_t)(r >> 15);
}
inline int16_t unzigzag16(uint32_t z) {
    return (int16_t)((z >> 1) ^ (~(z & 1) + 1));
}

struct Symbolized {
    std::vector<uint16_t> syms; // [0..MAX_SYM], ESC_SYM marks escape
    std::vector<int16_t>  esc;  // raw outliers in order
    size_t escapes = 0;
};

inline Symbolized symbolize_residuals(const std::vector<int16_t>& residuals) {
    Symbolized S;
    S.syms.resize(residuals.size());
    for (size_t i=0; i<residuals.size(); ++i) {
        uint32_t z = zigzag16(residuals[i]);
        if (z < MAX_SYM) {
            S.syms[i] = (uint16_t)z;
        } else {
            S.syms[i] = ESC_SYM;
            S.esc.push_back(residuals[i]);
            ++S.escapes;
        }
    }
    return S;
}
inline std::vector<int16_t> unsymbolize_residuals(const std::vector<uint16_t>& syms,
                                                  const std::vector<int16_t>& esc) {
    std::vector<int16_t> out; out.resize(syms.size());
    size_t ei=0;
    for (size_t i=0;i<syms.size();++i) {
        uint16_t s = syms[i];
        out[i] = (s==ESC_SYM) ? esc[ei++] : unzigzag16(s);
    }
    return out;
}

// ------------------------ Model (static) -------------------
struct Model {
    uint32_t L = RANS_L;
    std::vector<uint16_t> freq;     // size = ALPHABET
    std::vector<uint32_t> cdf;      // size = ALPHABET
    std::vector<uint16_t> lut_sym;  // size = L
};

inline Model build_model(const std::vector<uint16_t>& syms) {
    Model m;
    m.freq.assign(ALPHABET, 0);
    for (auto s: syms) m.freq[s]++;

    uint64_t total=0; for (auto f: m.freq) total+=f;
    if (total==0) { m.freq[0]=(uint16_t)m.L; total=m.L; }

    std::vector<double> prob(ALPHABET, 0.0);
    for (size_t s=0; s<ALPHABET; ++s)
        prob[s] = (double)m.freq[s] / (double)std::max<uint64_t>(1,total);

    uint32_t sum=0;
    for (size_t s=0; s<ALPHABET; ++s) {
        m.freq[s] = (uint16_t)std::max(1, (int)std::lround(prob[s]*m.L));
        sum += m.freq[s];
    }
    while (sum > m.L) {
        size_t idx = (size_t)std::distance(m.freq.begin(),
                      std::max_element(m.freq.begin(), m.freq.end()));
        if (m.freq[idx] > 1) { m.freq[idx]--; sum--; } else break;
    }
    while (sum < m.L) {
        size_t idx = (size_t)std::distance(m.freq.begin(),
                      std::min_element(m.freq.begin(), m.freq.end()));
        m.freq[idx]++; sum++;
    }

    m.cdf.resize(ALPHABET);
    uint32_t c=0; for (size_t s=0;s<ALPHABET;++s){ m.cdf[s]=c; c+=m.freq[s]; }

    m.lut_sym.resize(m.L);
    for (uint32_t s=0; s<ALPHABET; ++s) {
        uint32_t f=m.freq[s], start=m.cdf[s];
        for (uint32_t i=0;i<f;++i) m.lut_sym[start+i]=(uint16_t)s;
    }
    return m;
}

// -------------------------- rANS32 -------------------------
namespace rans32 {
    static constexpr int PREC = 12;            // log2(L)
    static constexpr uint32_t L  = RANS_L;     // 4096

    inline std::vector<uint8_t> encode(const std::vector<uint16_t>& syms,
                                       const Model& m)
    {
        std::vector<uint8_t> out;
        out.reserve(syms.size()/2 + 16);
        uint32_t x = 1u << 16;
        auto put = [&](uint8_t b){ out.push_back(b); };

        for (size_t i=syms.size(); i-- > 0; ) {
            uint16_t s  = syms[i];
            uint32_t f  = m.freq[s];
            uint32_t cf = m.cdf[s];

            while (x >= (f << (32 - PREC))) {
                put((uint8_t)(x & 0xFF));
                x >>= 8;
            }
            x = (x / f) * L + (x % f) + cf;
        }
        put((uint8_t)(x & 0xFF)); x >>= 8;
        put((uint8_t)(x & 0xFF)); x >>= 8;
        put((uint8_t)(x & 0xFF)); x >>= 8;
        put((uint8_t)(x & 0xFF));
        return out;
    }

    inline std::vector<uint16_t> decode(const std::vector<uint8_t>& in,
                                        size_t n_syms,
                                        const Model& m)
    {
        std::vector<uint16_t> out(n_syms);
        size_t ip = in.size();
        auto get = [&]()->uint32_t {
            if (ip==0) throw std::runtime_error("rANS underflow");
            return (uint32_t)in[--ip];
        };
        uint32_t x=0;
        x |= get(); x |= get()<<8; x |= get()<<16; x |= get()<<24;

        for (size_t i=0;i<n_syms;++i) {
            uint32_t slot = x & (L-1);
            uint16_t s    = m.lut_sym[slot];
            out[i] = s;

            uint32_t cf=m.cdf[s], f=m.freq[s];
            x = f * (x >> PREC) + (slot - cf);
            while (x < (1u<<16)) x = (x<<8) | get();
        }
        return out;
    }
}

// --------------------- file container I/O ------------------
struct Packed {
    int mode=0;       // 0: u8/RGB residuals, 1: s16 domain residuals
    int w=0,h=0,c=0;
    uint64_t n_syms=0;
    Model model;
    std::vector<uint8_t>  ans_bytes;
    std::vector<int16_t>  escapes;
};

inline void save_file(const std::string& path, const Packed& P) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("open write: " + path);

    uint32_t magic=FILE_MAGIC, L=P.model.L, ALPH=(uint32_t)P.model.freq.size();
    uint64_t esc_count=(uint64_t)P.escapes.size();
    uint64_t esc_bytes=esc_count*2;
    uint64_t ans_size =(uint64_t)P.ans_bytes.size();

    f.write((char*)&magic,4);
    f.write((char*)&P.mode,4);
    f.write((char*)&P.w,4); f.write((char*)&P.h,4); f.write((char*)&P.c,4);
    f.write((char*)&P.n_syms,8);
    f.write((char*)&L,4);
    f.write((char*)&ALPH,4);
    f.write((char*)P.model.freq.data(), (std::streamsize)(sizeof(uint16_t)*ALPH));
    f.write((char*)&esc_count,8);
    f.write((char*)&esc_bytes,8);
    f.write((char*)&ans_size,8);
    if (esc_bytes) f.write((char*)P.escapes.data(), (std::streamsize)esc_bytes);
    if (ans_size)  f.write((char*)P.ans_bytes.data(), (std::streamsize)ans_size);
    if (!f) throw std::runtime_error("write failed: " + path);
}

inline Packed load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("open read: " + path);
    Packed P;

    uint32_t magic; f.read((char*)&magic,4);
    if (magic != FILE_MAGIC) throw std::runtime_error("bad magic");

    f.read((char*)&P.mode,4);
    f.read((char*)&P.w,4); f.read((char*)&P.h,4); f.read((char*)&P.c,4);
    f.read((char*)&P.n_syms,8);

    uint32_t L, ALPH;
    f.read((char*)&L,4); f.read((char*)&ALPH,4);
    P.model.L = L;
    P.model.freq.resize(ALPH);
    f.read((char*)P.model.freq.data(), (std::streamsize)(sizeof(uint16_t)*ALPH));

    P.model.cdf.resize(ALPH);
    uint32_t cdf=0; for (uint32_t s=0;s<ALPH;++s){ P.model.cdf[s]=cdf; cdf+=P.model.freq[s]; }
    P.model.lut_sym.resize(L);
    for (uint32_t s=0;s<ALPH;++s) {
        uint32_t fsz=P.model.freq[s], start=P.model.cdf[s];
        for (uint32_t i=0;i<fsz;++i) P.model.lut_sym[start+i]=(uint16_t)s;
    }

    uint64_t esc_count, esc_bytes, ans_size;
    f.read((char*)&esc_count,8);
    f.read((char*)&esc_bytes,8);
    f.read((char*)&ans_size,8);

    P.escapes.resize(esc_count);
    if (esc_bytes) f.read((char*)P.escapes.data(), (std::streamsize)esc_bytes);

    P.ans_bytes.resize(ans_size);
    if (ans_size)  f.read((char*)P.ans_bytes.data(), (std::streamsize)ans_size);

    if (!f) throw std::runtime_error("read failed: " + path);
    return P;
}

// --------------- pack/unpack helpers ------------
struct Encoded {
    size_t escapes=0;
    size_t n_syms=0;
    size_t ans_bytes=0;
};

inline Encoded compress_to_file(const std::vector<int16_t>& residuals,
                                int mode, int w,int h,int c,
                                const std::string& outPath)
{
    Encoded info{};
    auto S = symbolize_residuals(residuals);
    auto M = build_model(S.syms);
    auto ans_bytes = rans32::encode(S.syms, M);

    Packed P;
    P.mode=mode; P.w=w; P.h=h; P.c=c;
    P.n_syms=(uint64_t)S.syms.size();
    P.model=std::move(M);
    P.ans_bytes=std::move(ans_bytes);
    P.escapes=std::move(S.esc);

    save_file(outPath, P);

    info.escapes=P.escapes.size();
    info.n_syms =(size_t)P.n_syms;
    info.ans_bytes=P.ans_bytes.size();
    return info;
}

inline std::vector<int16_t> decompress_file(const std::string& inPath) {
    Packed P = load_file(inPath);
    auto syms = rans32::decode(P.ans_bytes, (size_t)P.n_syms, P.model);
    return unsymbolize_residuals(syms, P.escapes);
}

} // namespace ans
#endif //BYTE2BITPROJECT1_ANSRESIDUAL_H