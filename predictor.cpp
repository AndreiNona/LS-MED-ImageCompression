#include "predictor.h"
#include <algorithm>
#include <cmath>
#include <iostream>

//Hook for printing stats in main.cpp
LsBreakdown g_last_ls_breakdown;

// ---------- MED predictor(fallback predictor) ----------
int med_predict(int A, int B, int C) {
    if (C >= std::max(A, B)) return std::min(A, B);
    if (C <= std::min(A, B)) return std::max(A, B);
    return A + B - C;
}

// ---------- uint8 path ----------
static inline int get_px_u8(const Image& im, int x, int y, int ch) {
    if (x < 0 || y < 0 || x >= im.w || y >= im.h) return 0; // zero border
    return im.px[(y*im.w + x)*im.c + ch];
}

// ---------- int16 (RCT) path ----------
static inline int get_px_s16(const Image16& im, int x, int y, int ch) {
    if (x < 0 || y < 0 || x >= im.w || y >= im.h) return 0;
    return im.px[(y*im.w + x)*im.c + ch];
}

static inline void apply_ridge(std::vector<double>& ATA, int N, double lambda) {
    if (lambda <= 0.0) return;
    for (int i = 0; i < N; ++i) ATA[i*N + i] += lambda;
}
std::vector<int16_t> compute_residuals_MED_u8(const Image& src) {
    std::vector<int16_t> res(static_cast<size_t>(src.w)*src.h*src.c);
    for (int y=0; y<src.h; ++y)

        for (int x=0; x<src.w; ++x)

            for (int ch=0; ch<src.c; ++ch) {

                int A = get_px_u8(src, x-1, y,   ch);
                int B = get_px_u8(src, x,   y-1, ch);
                int C = get_px_u8(src, x-1, y-1, ch);
                int pred = med_predict(A,B,C);
                int actual = get_px_u8(src, x, y, ch);
                res[(size_t)(y*src.w + x)*src.c + ch] =
                    (int16_t)(actual - pred);

            }
    return res;
}

Image reconstruct_from_residuals_MED(const std::vector<int16_t>& residuals,
                                     const Image& shape) {
    Image rec; rec.w=shape.w; rec.h=shape.h; rec.c=shape.c;
    rec.px.resize(static_cast<size_t>(rec.w)*rec.h*rec.c, 0);
    for (int y=0; y<rec.h; ++y)

        for (int x=0; x<rec.w; ++x)

            for (int ch=0; ch<rec.c; ++ch) {

                int A = get_px_u8(rec, x-1, y,   ch);
                int B = get_px_u8(rec, x,   y-1, ch);
                int C = get_px_u8(rec, x-1, y-1, ch);
                int pred = med_predict(A,B,C);
                int16_t r = residuals[(size_t)(y*rec.w + x)*rec.c + ch];
                int val = pred + (int)r;
                rec.px[(size_t)(y*rec.w + x)*rec.c + ch] =
                    (unsigned char)std::clamp(val, 0, 255);

            }
    return rec;
}

std::vector<int16_t> compute_residuals_MED_s16(const Image16& src) {
    std::vector<int16_t> res(static_cast<size_t>(src.w)*src.h*src.c);

    for (int y=0; y<src.h; ++y)

        for (int x=0; x<src.w; ++x)

            for (int ch=0; ch<src.c; ++ch) {

                int A = get_px_s16(src, x-1, y,   ch);
                int B = get_px_s16(src, x,   y-1, ch);
                int C = get_px_s16(src, x-1, y-1, ch);
                int pred = med_predict(A,B,C);
                int actual = get_px_s16(src, x, y, ch);
                res[(size_t)(y*src.w + x)*src.c + ch] =
                    (int16_t)(actual - pred);
            }
    return res;
}

Image16 reconstruct_from_residuals_MED_s16(const std::vector<int16_t>& residuals,
                                           const Image16& shape) {
    Image16 rec; rec.w=shape.w; rec.h=shape.h; rec.c=shape.c;
    rec.px.resize(static_cast<size_t>(rec.w)*rec.h*rec.c, 0);

    for (int y=0; y<rec.h; ++y)

        for (int x=0; x<rec.w; ++x)

            for (int ch=0; ch<rec.c; ++ch) {

                int A = get_px_s16(rec, x-1, y,   ch);
                int B = get_px_s16(rec, x,   y-1, ch);
                int C = get_px_s16(rec, x-1, y-1, ch);
                int pred = med_predict(A,B,C);
                int16_t r = residuals[(size_t)(y*rec.w + x)*rec.c + ch];
                rec.px[(size_t)(y*rec.w + x)*rec.c + ch] =
                    (int16_t)(pred + (int)r);
            }
    return rec;
}

// ---------- residual visualization for uint8 ----------
static inline unsigned char clamp8_vis(int v) {
    return (unsigned char)std::min(255, std::max(0, v));
}

static inline unsigned char clamp8_vis(int v); // already present below for u8


// ================= LS / Gaussian Jordan predictor ===================

// return:  false : failed / true  : success
static bool gauss_solve(std::vector<double>& A, std::vector<double>& b, int n, double lambda ) {
    if (n <= 0) return false;
    if (lambda != 0.0) {
        for (int d = 0; d < n; ++d) A[d*n + d] += lambda;
    }
    const double eps = 1e-12; // singularity guard

    for (int k = 0; k < n; ++k) {
        // ---- pivot: pick row with largest |A[i,k]| ----
        int piv = k;
        double best = std::fabs(A[k*n + k]);
        for (int i = k + 1; i < n; ++i) {
            double v = std::fabs(A[i*n + k]);
            if (v > best) { best = v; piv = i; }
        }
        if (best < eps) return false;

        // swap row k and piv
        if (piv != k) {
            for (int j = 0; j < n; ++j) std::swap(A[k*n + j], A[piv*n + j]);
            std::swap(b[k], b[piv]);
        }

        // ---- normalize pivot row (pivot = 1) ----
        double Akk = A[k*n + k];
        if (std::fabs(Akk) < eps) return false;
        double inv = 1.0 / Akk;

        A[k*n + k] = 1.0;
        for (int j = 0; j < n; ++j) if (j != k) A[k*n + j] *= inv;
        b[k] *= inv;

        // ---- eliminate this column from all other rows ----
        for (int i = 0; i < n; ++i) if (i != k) {
            double f = A[i*n + k];
            if (f == 0.0) continue;
            A[i*n + k] = 0.0;
            // subtract f * (pivot row) from row i
            for (int j = 0; j < n; ++j) if (j != k) A[i*n + j] -= f * A[k*n + j];
            b[i] -= f * b[k];
        }
    }
    return true;
}

// Choose a neighbor vector of length N
// If N<4 we take the first N ; if border invalid, we report false.
template <typename PixelGetter>
static bool build_neighbor_vec(int x, int y, int ch, int N,
                               const PixelGetter& get,
                               std::vector<double>& nvec) {
    nvec.assign(N, 0.0);
    int xs[4] = { x-1, x,   x-1, x+1 };
    int ys[4] = { y,   y-1, y-1, y-1 };
    for (int i=0; i<N; ++i) {

        int xi = xs[i], yi = ys[i];
        if (xi < 0 || yi < 0 || get.width()<=xi || get.height()<=yi) return false; //false will eventually trigger MED fallback
        nvec[i] = (double)get(xi, yi, ch);

    }
    return true;
}


template <typename PixelGetter>
static int accumulate_window_normal_eq(int x, int y, int ch, int N,
                                       int winW, int winH,
                                       const PixelGetter& get,
                                       std::vector<double>& ATA,
                                       std::vector<double>& ATy) {
    ATA.assign(N*N, 0.0);
    ATy.assign(N,   0.0);
    std::vector<double> v(N);
    int count = 0;

    // window rows: [y - winH, y]
    int yStart = std::max(0, y - winH);
    for (int yy = yStart; yy <= y; ++yy) {


        int xStart = std::max(0, x - winW);
        int xEnd   = std::min(get.width()-1, x-1);

        for (int xx = xStart; xx <= xEnd; ++xx) {

            if (!build_neighbor_vec(xx, yy, ch, N, get, v)) continue;

            auto tgt = (double)get(xx, yy, ch);


            for (int i=0; i<N; ++i) {

                ATy[i] += v[i] * tgt;
                double vi = v[i];
                for (int j=0; j<N; ++j) ATA[i*N + j] += vi * v[j];

            }
            ++count;
        }
    }
    return count;
}

// -------------------- u8 path (RGB/Gray) --------------------
struct GetterU8 {
    const Image& im;
    [[nodiscard]] int width() const { return im.w; }
    [[nodiscard]] int height() const { return im.h; }
    int operator()(int x, int y, int ch) const {
        return im.px[(y*im.w + x)*im.c + ch];
    }
};
// samples >= N+2 -> gauss solve
std::vector<int16_t> compute_residuals_LS_u8(const Image& src, int N, int winW, int winH) {
    std::vector<int16_t> res((size_t)src.w*src.h*src.c);


    Image ctx = src;
    ctx.px.assign((size_t)ctx.w*ctx.h*ctx.c, 0);

    GetterU8 getCtx{ctx};
    std::vector<double> ATA, ATy, nvec;

    size_t ls_count = 0, med_count = 0;

    for (int y=0; y<src.h; ++y) {
        for (int x=0; x<src.w; ++x) {
            for (int ch=0; ch<src.c; ++ch) {


                bool ls_ok = false;
                int pred = 0;

                int samples = accumulate_window_normal_eq(x, y, ch, N, winW, winH, getCtx, ATA, ATy);
                if (samples >= N + 2) {

                    std::vector<double> w = ATy;


                    if (gauss_solve(ATA, w, N, 1e-3) && build_neighbor_vec(x, y, ch, N, getCtx, nvec)) {

                        double p = 0.0; for (int i=0;i<N;++i) p += w[i]*nvec[i];
                        pred = std::clamp((int)std::llround(p), 0, 255);
                        ls_ok = true; ++ls_count;

                    }
                }

                if (!ls_ok) {

                    int A = (x-1>=0) ? getCtx(x-1,y,ch) : 0;
                    int B = (y-1>=0) ? getCtx(x,y-1,ch) : 0;
                    int C = (x-1>=0 && y-1>=0) ? getCtx(x-1,y-1,ch) : 0;
                    pred = med_predict(A,B,C);

                    ++med_count;

                }

                int actual = (int)src.px[(size_t)(y*src.w + x)*src.c + ch];
                auto r = (int16_t)(actual - pred);
                res[(size_t)(y*src.w + x)*src.c + ch] = r;

                // Update context exactly like the decoder will
                int recon = std::clamp(pred + (int)r, 0, 255);
                ctx.px[(size_t)(y*ctx.w + x)*ctx.c + ch] = (unsigned char)recon;
            }
        }
    }

    std::cout << "Prediction stats: LS=" << ls_count
              << " MED=" << med_count
              << " Total=" << (size_t)src.w*src.h*src.c
              << " (" << (100.0*ls_count/((size_t)src.w*src.h*src.c)) << "% LS)\n";

    //Hook for printing stats in main.cpp
    g_last_ls_breakdown.used_ls  = ls_count;
    g_last_ls_breakdown.used_med = med_count;
    return res;
}



Image reconstruct_from_residuals_LS_u8(const std::vector<int16_t>& residuals,
                                       const Image& shape, int N, int winW, int winH) {
    Image rec = shape;
    rec.px.assign((size_t)rec.w * rec.h * rec.c, 0);

    GetterU8 get{rec};
    std::vector<double> ATA, ATy, nvec;

    for (int y = 0; y < rec.h; ++y) {

        for (int x = 0; x < rec.w; ++x) {

            for (int ch = 0; ch < rec.c; ++ch) {

                bool ls_ok = false;
                int pred = 0;

                int samples = accumulate_window_normal_eq(x, y, ch, N, winW, winH, get, ATA, ATy);
                if (samples >= N + 2) {

                    std::vector<double> w = ATy; // solve A w = b

                    if (gauss_solve(ATA, w, N, 1e-3) && build_neighbor_vec(x, y, ch, N, get, nvec)) {

                        double p = 0.0; for (int i = 0; i < N; ++i) p += w[i] * nvec[i];
                        pred = std::clamp((int)std::llround(p), 0, 255);
                        ls_ok = true;
                    }
                }

                if (!ls_ok) {
                    pred = med_predict(
                        (x-1>=0 ? get(x-1,y,ch) : 0),
                        (y-1>=0 ? get(x,y-1,ch) : 0),
                        (x-1>=0 && y-1>=0 ? get(x-1,y-1,ch) : 0)
                    );
                }

                int16_t r = residuals[(size_t)(y*rec.w + x)*rec.c + ch];
                int val = pred + (int)r;
                rec.px[(size_t)(y*rec.w + x)*rec.c + ch] = (unsigned char)std::clamp(val, 0, 255);
            }
        }
    }
    return rec;
}



// -------------------- s16 path (RCT) --------------------
struct GetterS16 {

    const Image16& im;
    int width() const { return im.w; }
    int height() const { return im.h; }
    int operator()(int x, int y, int ch) const {

        return im.px[(y*im.w + x)*im.c + ch];

    }
};

//Same logic as u8 but no clamping
std::vector<int16_t> compute_residuals_LS_s16(const Image16& src, int N, int winW, int winH) {
    std::vector<int16_t> res((size_t)src.w*src.h*src.c);

    Image16 ctx = src;
    ctx.px.assign((size_t)ctx.w*ctx.h*ctx.c, 0);

    GetterS16 getCtx{ctx};
    std::vector<double> ATA, ATy, nvec;

    size_t ls_count = 0, med_count = 0;

    for (int y=0; y<src.h; ++y) {

        for (int x=0; x<src.w; ++x) {

            for (int ch=0; ch<src.c; ++ch) {

                bool ls_ok = false;
                int pred = 0;

                int samples = accumulate_window_normal_eq(x, y, ch, N, winW, winH, getCtx, ATA, ATy);
                if (samples >= N + 2) {

                    std::vector<double> w = ATy;

                    if (gauss_solve(ATA, w, N, 1e-3) && build_neighbor_vec(x, y, ch, N, getCtx, nvec)) {

                        double p = 0.0; for (int i=0;i<N;++i) p += w[i]*nvec[i];
                        pred = (int)std::llround(p);   // s16 path: no clamp
                        ls_ok = true; ++ls_count;
                    }
                }

                if (!ls_ok) {

                    int A = (x-1>=0) ? getCtx(x-1,y,ch) : 0;
                    int B = (y-1>=0) ? getCtx(x,y-1,ch) : 0;
                    int C = (x-1>=0 && y-1>=0) ? getCtx(x-1,y-1,ch) : 0;
                    pred = med_predict(A,B,C);
                    ++med_count;

                }

                int actual = (int)src.px[(size_t)(y*src.w + x)*src.c + ch];
                int16_t r = (int16_t)(actual - pred);
                res[(size_t)(y*src.w + x)*src.c + ch] = r;

                int recon = pred + (int)r;       // s16: keep signed
                ctx.px[(size_t)(y*ctx.w + x)*ctx.c + ch] = (int16_t)recon;
            }
        }
    }

    std::cout << "Prediction stats: LS=" << ls_count
              << " MED=" << med_count
              << " Total=" << (size_t)src.w*src.h*src.c
              << " (" << (100.0*ls_count/((size_t)src.w*src.h*src.c)) << "% LS)\n";
    //Hook for printing stats in main.cpp
    g_last_ls_breakdown.used_ls  = ls_count;
    g_last_ls_breakdown.used_med = med_count;
    return res;
}


//Same logic as u8
Image16 reconstruct_from_residuals_LS_s16(const std::vector<int16_t>& residuals,
                                          const Image16& shape, int N, int winW, int winH) {
    Image16 rec = shape;
    rec.px.assign((size_t)rec.w * rec.h * rec.c, 0);

    GetterS16 get{rec};
    std::vector<double> ATA, ATy, nvec;

    for (int y = 0; y < rec.h; ++y) {
        for (int x = 0; x < rec.w; ++x) {
            for (int ch = 0; ch < rec.c; ++ch) {
                bool ls_ok = false;
                int pred = 0;

                int samples = accumulate_window_normal_eq(x, y, ch, N, winW, winH, get, ATA, ATy);
                if (samples >= N + 2) {
                    std::vector<double> w = ATy;
                    if (gauss_solve(ATA, w, N, 1e-3) && build_neighbor_vec(x, y, ch, N, get, nvec)) {
                        double p = 0.0; for (int i = 0; i < N; ++i) p += w[i] * nvec[i];
                        pred = (int)std::llround(p); // int16 domain, no clamp here
                        ls_ok = true;
                    }
                }

                if (!ls_ok) {
                    pred = med_predict(
                        (x-1>=0 ? get(x-1,y,ch) : 0),
                        (y-1>=0 ? get(x,y-1,ch) : 0),
                        (x-1>=0 && y-1>=0 ? get(x-1,y-1,ch) : 0)
                    );
                }

                int16_t r = residuals[(size_t)(y*rec.w + x)*rec.c + ch];
                rec.px[(size_t)(y*rec.w + x)*rec.c + ch] = (int16_t)(pred + (int)r);
            }
        }
    }
    return rec;
}


// -------- visualisation  --------
//Only used for testing
Image residuals_visual_rgb8(const std::vector<int16_t>& residuals, const Image& shape) {
    Image vis; vis.w=shape.w; vis.h=shape.h; vis.c=shape.c;
    vis.px.resize(static_cast<size_t>(vis.w)*vis.h*vis.c);
    for (int i=0; i<vis.w*vis.h*vis.c; ++i) {
        int v = 128 + (int)residuals[i]; // map 0 residual -> 128 mid-gray
        vis.px[i] = clamp8_vis(v);
    }
    return vis;
}
Image residuals_visual_s16(const std::vector<int16_t>& residuals, const Image16& shape) {
    Image vis; vis.w = shape.w; vis.h = shape.h; vis.c = shape.c; // 3 channels
    vis.px.resize(static_cast<size_t>(vis.w) * vis.h * vis.c);
    for (size_t i = 0; i < vis.px.size(); ++i) {
        int v = 128 + (int)residuals[i];    // 0 residual -> 128 mid-gray
        vis.px[i] = clamp8_vis(v);
    }
    return vis;
}