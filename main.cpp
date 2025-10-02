#include "imageIO.h"
#include "predictor.h"
#include "residualIO.h"
#include "ansResidual.h"

#include <iostream>
#include <chrono>
#include <string>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <cctype>
#include <numeric>
#include <iomanip>
#include <fstream>

namespace fs = std::filesystem;

// --- helpers ---
static uint64_t file_size_bytes(const std::string& path) {
    std::error_code ec;
    auto sz = fs::file_size(path, ec);
    return ec ? 0ull : (uint64_t)sz;
}

//Filter image files
static bool has_ext_ci(const fs::path& p, std::initializer_list<const char*> exts) {
    auto ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char ch){ return (char)std::tolower(ch); });
    for (auto e : exts) {
        std::string s = e;
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char ch){ return (char)std::tolower(ch); });
        if (ext == s) return true;
    }
    return false;
}

static std::string stem_of(const fs::path& p) {
    return p.stem().string();
}

static fs::path with_suffix_and_same_ext(const fs::path& inPath,
                                         const fs::path& outDir,
                                         const std::string& suffix)
{
    //Example: cat.png + "_reconstructed" = cat_reconstructed.png
    fs::path out = outDir / (stem_of(inPath) + suffix + inPath.extension().string());
    return out;
}
//Sane as with_suffix_and_same_ext but with PNG for visualisation
static fs::path with_suffix_png(const fs::path& inPath,
                                const fs::path& outDir,
                                const std::string& suffix_no_ext)
{
    fs::path out = outDir / (stem_of(inPath) + suffix_no_ext + ".png");
    return out;
}
//Sane as with_suffix_and_same_ext but with arbitrary extension
static fs::path with_suffix_ext(const fs::path& inPath,
                                const fs::path& outDir,
                                const std::string& suffix,
                                const std::string& ext)
{
    fs::path out = outDir / (stem_of(inPath) + suffix + ext);
    return out;
}
//In case we don't have a target dir
static void ensure_dir(const fs::path& dir) {
    std::error_code ec;
    if (!dir.empty() && !fs::exists(dir, ec)) {
        fs::create_directories(dir, ec);
        if (ec) throw std::runtime_error("Failed to create out-dir: " + dir.string());
    }
}

static void print_ans_report(const char* tag,
                             const std::string& path,
                             int w,int h,int c)
{
    uint64_t comp = file_size_bytes(path);
    if (comp == 0) {
        std::cout << tag << " failed to stat file: " << path << "\n";
        return;
    }
    const uint64_t pixels = (uint64_t)w*h*c;
    const double bpp = (8.0 * comp) / (double)pixels;
    const double ratio_vs_resid = (double)comp / (double)(pixels * 2ull);
    const double ratio_vs_rgb   = (double)comp / (double)(w*h*3ull);

    std::cout << tag << "  file=" << path
              << "  size=" << comp << " bytes"
              << "  bpp=" << bpp
              << "  ratio_vs_residual=" << ratio_vs_resid
              << "  ratio_vs_rawRGB=" << ratio_vs_rgb << "\n";
}

// Get input files (single or directory)
// in: Path / Out: Sorted list of input image paths
static std::vector<fs::path> collect_inputs(const fs::path& inPath, bool recursive) {
    std::vector<fs::path> files;
    std::error_code ec;
    if (fs::is_regular_file(inPath, ec)) {
        files.push_back(inPath);
        return files;
    }
    if (!fs::is_directory(inPath, ec)) {
        throw std::runtime_error("Input path is neither a file nor a directory: " + inPath.string());
    }

    auto add_if_image = [&](const fs::directory_entry& de){
        const auto& p = de.path();
        if (!de.is_regular_file()) return;
        if (has_ext_ci(p, {".png",".jpg",".jpeg",".bmp",".tga",".ppm",".pgm",".pnm"})) {
            files.push_back(p);
        }
    };

    if (recursive) {
        for (auto it = fs::recursive_directory_iterator(inPath); it != fs::recursive_directory_iterator(); ++it) {
            std::error_code fec;
            if (it->is_regular_file(fec)) {
                add_if_image(*it);
            }
        }
    } else {
        for (auto& de : fs::directory_iterator(inPath)) {
            add_if_image(de);
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

struct Stats {
    std::string file;         // input file name (no path)
    std::string mode;         // rgb | yuv | ls(rgb) | ls(yuv)
    int w=0, h=0, c=0;
    uint64_t pixels=0;
    std::string fmt;          // PNG/JPG/PPM/PGM/etc...
    uint64_t orig_bytes=0;    // input file size
    uint64_t ans_bytes=0;     // compressed residual file size
    double   bpp=0.0;
    double   ratio_vs_resid=0.0;
    double   ratio_vs_rawrgb=0.0;
    int64_t  t_io_ms=0, t_pred_ms=0, t_rec_ms=0;
    double   thr_pred_mpps=0.0; // megapixels/s during predict
    double   thr_rec_mpps=0.0;  // megapixels/s during reconstruct
    bool     equal=false;

    //  LS statistics
    long long ls_count = -1;
    long long med_count = -1;
    double    ls_pct = std::numeric_limits<double>::quiet_NaN();
};

static void write_batch_summary(const std::filesystem::path& outDir,
                                const std::vector<Stats>& all)
{
    using namespace std;
    namespace fs = std::filesystem;

    fs::path out = outDir / "batch_summary.txt";
    ofstream ofs(out);
    if (!ofs) throw runtime_error("Failed to open summary file: " + out.string());

    // ---- header ----
    ofs << "Batch summary (" << all.size() << " image(s))\n\n";

    // ---- per-image table ----
    ofs << left
        << setw(22) << "file"
        << setw(9)  << "mode"
        << setw(12) << "WxHxC"
        << setw(7)  << "fmt"
        << setw(10) << "pixels"
        << setw(12) << "orig_B"
        << setw(12) << "ANS_B"
        << setw(9)  << "bpp"
        << setw(11) << "r_vs_resid"
        << setw(11) << "r_vs_RGB"
        << setw(9)  << "IOms"
        << setw(9)  << "Predms"
        << setw(9)  << "Recms"
        << setw(10) << "MPix/sP"
        << setw(10) << "MPix/sR"
        << setw(7)  << "Equal"
        << setw(12) << "LS"
        << setw(12) << "MED"
        << setw(8)  << "%LS"
        << "\n";

    for (const auto& s : all) {
        ofs << left
            << setw(22) << s.file
            << setw(9)  << s.mode
            << setw(12) << (to_string(s.w)+"x"+to_string(s.h)+"x"+to_string(s.c))
            << setw(7)  << s.fmt
            << setw(10) << s.pixels
            << setw(12) << s.orig_bytes
            << setw(12) << s.ans_bytes
            << setw(9)  << fixed << setprecision(3) << s.bpp
            << setw(11) << fixed << setprecision(6) << s.ratio_vs_resid
            << setw(11) << fixed << setprecision(6) << s.ratio_vs_rawrgb
            << setw(9)  << s.t_io_ms
            << setw(9)  << s.t_pred_ms
            << setw(9)  << s.t_rec_ms
            << setw(10) << fixed << setprecision(2) << s.thr_pred_mpps
            << setw(10) << fixed << setprecision(2) << s.thr_rec_mpps
            << setw(7)  << (s.equal ? "YES" : "NO");

        if (s.ls_count >= 0 && s.med_count >= 0) {
            ofs << setw(12) << s.ls_count
                << setw(12) << s.med_count
                << setw(8)  << fixed << setprecision(4) << s.ls_pct;
        } else {
            ofs << setw(12) << "n/a"
                << setw(12) << "n/a"
                << setw(8)  << "n/a";
        }
        ofs << "\n";
    }

    // ---- statistics ----
    if (!all.empty()) {
        uint64_t sum_pixels = 0, sum_ans = 0, sum_orig = 0;
        uint64_t pass_equal = 0;
        long long sum_io=0, sum_pred=0, sum_rec=0;

        for (const auto& s : all) {
            sum_pixels += s.pixels;
            sum_ans    += s.ans_bytes;
            sum_orig   += s.orig_bytes;
            sum_io     += s.t_io_ms;
            sum_pred   += s.t_pred_ms;
            sum_rec    += s.t_rec_ms;
            pass_equal += s.equal ? 1 : 0;
        }
        // weighted bpp (by pixels)
        double bpp_weighted = sum_pixels ? (8.0 * (double)sum_ans) / (double)sum_pixels : 0.0;
        // overall throughput (MPix/s) using sums
        double mpix_total = sum_pixels / 1e6;
        double thr_pred = (sum_pred>0) ? (1000.0 * mpix_total / (double)sum_pred) : 0.0;
        double thr_rec  = (sum_rec>0)  ? (1000.0 * mpix_total / (double)sum_rec)  : 0.0;

        ofs << "\n--- Totals ---\n";
        ofs << "images: " << all.size() << "\n";
        ofs << "pixels total: " << sum_pixels << "\n";
        ofs << "orig bytes total: " << sum_orig << "\n";
        ofs << "ANS bytes total: " << sum_ans << "\n";
        ofs << "weighted bpp: " << fixed << setprecision(3) << bpp_weighted << "\n";
        ofs << "avg IO ms/img: "   << (sum_io   / (long long)all.size()) << "\n";
        ofs << "avg Pred ms/img: " << (sum_pred / (long long)all.size()) << "\n";
        ofs << "avg Rec ms/img: "  << (sum_rec  / (long long)all.size()) << "\n";
        ofs << "overall Pred throughput (MPix/s): " << fixed << setprecision(2) << thr_pred << "\n";
        ofs << "overall Rec throughput (MPix/s): "  << fixed << setprecision(2) << thr_rec  << "\n";
        ofs << "equality pass: " << pass_equal << " / " << all.size() << "\n";
    }

    ofs.close();
    std::cout << "Wrote summary: " << out.string() << "\n";
}

int main(int argc, char** argv) {
try {
    using namespace std::chrono;

    // -------- env helpers  --------
    auto env_str = [](const char* k, const std::string& def = std::string()) {
        if (const char* v = std::getenv(k)) return std::string(v);
        return def;
    };
    auto env_int = [](const char* k, int def) {
        if (const char* v = std::getenv(k)) return std::atoi(v);
        return def;
    };
    auto env_bool = [](const char* k, bool def) {
        if (const char* v = std::getenv(k)) {
            std::string s(v);
            std::transform(s.begin(), s.end(), s.begin(), ::tolower);
            return (s=="1" || s=="true" || s=="yes" || s=="on");
        }
        return def;
    };
    auto lower = [](std::string s){ for (auto& c: s) c = (char)std::tolower((unsigned char)c); return s; };

    // -------- read config --------
    bool IMG_COMPARE_YUV      = env_bool("IMG_COMPARE_YUV", false);
    bool IMG_COMPARE_SAVE_VIS = env_bool("IMG_COMPARE_SAVE_VIS", false);
    std::string IMG_COMPARE_SUFFIX = env_str("IMG_COMPARE_SUFFIX", "_cmp");

    fs::path inPath = env_str("IMG_IN", "test_images/test.png");
    fs::path outDir = env_str("IMG_OUT_DIR", ".");
    bool recursive  = env_bool("IMG_RECURSIVE", false);

    std::string mode = lower(env_str("IMG_MODE", "rgb"));          // rgb | yuv | ls
    std::string lsOn = lower(env_str("IMG_LS_ON", "rgb"));         // rgb | yuv
    if (mode == "rct") mode = "yuv";

    // LS parameters
    int N           = env_int("IMG_LS_N", 4);
    int winW        = env_int("IMG_LS_WIN_W", 4);
    int winH        = env_int("IMG_LS_WIN_H", 4);

    bool saveVis    = env_bool("IMG_SAVE_RES_VIS", false);
    std::string saveResPath = env_str("IMG_SAVE_RES", "");
    std::string loadResPath = env_str("IMG_LOAD_RES", "");

    // --------  single file residual load --------
    if (!loadResPath.empty()) {
        ensure_dir(outDir);
        auto rf = load_residuals(loadResPath);
        if (rf.mode == 0) {
            Image shape; shape.w = rf.w; shape.h = rf.h; shape.c = rf.c;
            auto t0 = high_resolution_clock::now();
            Image rec = reconstruct_from_residuals_MED(rf.residuals, shape);
            auto t1 = high_resolution_clock::now();
            fs::path out = outDir / "out_reconstructed_from_file.png";
            save_png(out.string(), rec);
            std::cout << "[FROM FILE] mode=RGB  " << rf.w << "x" << rf.h << "x" << rf.c
                      << " | Reconstruct: " << duration_cast<milliseconds>(t1 - t0).count() << " ms\n";
        } else {
            Image16 shape; shape.w = rf.w; shape.h = rf.h; shape.c = rf.c;
            auto t0 = high_resolution_clock::now();
            Image16 yuv_rec = reconstruct_from_residuals_MED_s16(rf.residuals, shape);
            Image rec = yuv_to_rgb(yuv_rec);
            auto t1 = high_resolution_clock::now();
            fs::path out = outDir / "out_reconstructed_from_file.png";
            save_png(out.string(), rec);
            std::cout << "[FROM FILE] mode=yuv  " << rf.w << "x" << rf.h << "x" << rf.c
                      << " | Reconstruct: " << duration_cast<milliseconds>(t1 - t0).count() << " ms\n";
        }
        return 0;
    }

    std::vector<fs::path> inputs = collect_inputs(inPath, recursive);
    if (inputs.empty()) {
        std::cerr << "No input images found in: " << inPath << "\n";
        return 2;
    }
    ensure_dir(outDir);

    std::vector<Stats> allStats;

    for (const auto& path : inputs) {
    try {
        auto tLoad0 = std::chrono::high_resolution_clock::now();
        Image rgb = load_image(path.string());
        auto tLoad1 = std::chrono::high_resolution_clock::now();

        // start stats
        Stats st;
        st.file   = path.filename().string();
        st.mode   = (mode == "ls") ? std::string("ls(") + lsOn + ")" : mode;
        st.w = rgb.w; st.h = rgb.h; st.c = rgb.c;
        st.pixels = (uint64_t)rgb.w * rgb.h * rgb.c;
        st.orig_bytes = file_size_bytes(path.string());
        st.fmt = rgb.format == ImageFormat::PNG ? "PNG" :
                 rgb.format == ImageFormat::JPG ? "JPG" :
                 rgb.format == ImageFormat::BMP ? "BMP" :
                 rgb.format == ImageFormat::TGA ? "TGA" :
                 rgb.format == ImageFormat::PPM ? "PPM" :
                 rgb.format == ImageFormat::PGM ? "PGM" : "UNK";
        st.t_io_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tLoad1 - tLoad0).count();

        if (IMG_COMPARE_YUV) {
            if (rgb.c != 3) {
                std::cout << "[COMPARE] Skipping non-RGB image: "
                          << path.filename().string() << " (c=" << rgb.c << ")\n";
                continue;
            }

            const uint64_t pixels = (uint64_t)rgb.w * rgb.h * rgb.c;

            // ===== RGB → LS =====
            auto tPred0 = std::chrono::high_resolution_clock::now();
            auto resid_rgb = compute_residuals_LS_u8(rgb, N, winW, winH);
            auto tPred1 = std::chrono::high_resolution_clock::now();

            if (IMG_COMPARE_SAVE_VIS) {
                auto vis = residuals_visual_rgb8(resid_rgb, rgb);
                save_png(with_suffix_png(path, outDir, IMG_COMPARE_SUFFIX + "_rgb_residuals_vis").string(), vis);
            }

            auto ans_rgb = with_suffix_ext(path, outDir, IMG_COMPARE_SUFFIX + "_rgb", ".r16ans");
            ans::compress_to_file(resid_rgb, /*mode=*/0, rgb.w, rgb.h, rgb.c, ans_rgb.string());

            auto rec_rgb = reconstruct_from_residuals_LS_u8(resid_rgb, rgb, N, winW, winH);
            auto tRec1 = std::chrono::high_resolution_clock::now();

            rec_rgb.format = rgb.format; // ensure save_image picks the right writer
            save_image(with_suffix_and_same_ext(path, outDir, IMG_COMPARE_SUFFIX + "_rgb_reconstructed").string(), rec_rgb);

            const uint64_t ansB_rgb = file_size_bytes(ans_rgb.string());
            const double   bpp_rgb  = pixels ? (8.0 * (double)ansB_rgb) / (double)pixels : 0.0;
            const long long pred_ms_rgb = std::chrono::duration_cast<std::chrono::milliseconds>(tPred1 - tPred0).count();
            const long long rec_ms_rgb  = std::chrono::duration_cast<std::chrono::milliseconds>(tRec1  - tPred1).count();
            const bool equal_rgb = images_equal(rgb, rec_rgb);

            // ===== yuv → LS =====
            Image16 yuv = rgb_to_yuv(rgb);

            auto tPred0y = std::chrono::high_resolution_clock::now();
            auto resid_yuv = compute_residuals_LS_s16(yuv, N, winW, winH);
            auto tPred1y = std::chrono::high_resolution_clock::now();

            if (IMG_COMPARE_SAVE_VIS) {
                auto vis = residuals_visual_s16(resid_yuv, yuv);
                save_png(with_suffix_png(path, outDir, IMG_COMPARE_SUFFIX + "_yuv_residuals_vis").string(), vis);
            }

            auto ans_yuv = with_suffix_ext(path, outDir, IMG_COMPARE_SUFFIX + "_yuv", ".r16ans");
            ans::compress_to_file(resid_yuv, /*mode=*/1, yuv.w, yuv.h, yuv.c, ans_yuv.string());

            auto yuv_rec16 = reconstruct_from_residuals_LS_s16(resid_yuv, yuv, N, winW, winH);
            Image rec_yuv = yuv_to_rgb(yuv_rec16);
            auto tRec1y = std::chrono::high_resolution_clock::now();

            rec_yuv.format = rgb.format;
            save_image(with_suffix_and_same_ext(path, outDir, IMG_COMPARE_SUFFIX + "_yuv_reconstructed").string(), rec_yuv);

            const uint64_t ansB_yuv = file_size_bytes(ans_yuv.string());
            const double   bpp_yuv  = pixels ? (8.0 * (double)ansB_yuv) / (double)pixels : 0.0;
            const long long pred_ms_yuv = std::chrono::duration_cast<std::chrono::milliseconds>(tPred1y - tPred0y).count();
            const long long rec_ms_yuv  = std::chrono::duration_cast<std::chrono::milliseconds>(tRec1y  - tPred1y).count();
            const bool equal_yuv = images_equal(rgb, rec_yuv);

            std::cout << std::fixed << std::setprecision(6);

            std::cout << "[COMPARE][RGB]  "  << path.filename().string()
                      << "  ansB=" << ansB_rgb
                      << "  bpp="  << bpp_rgb
                      << "  Equal=" << (equal_rgb ? "YES" : "NO")
                      << "  Pred=" << pred_ms_rgb << "ms"
                      << "  Rec="  << rec_ms_rgb  << "ms\n";

            std::cout << "[COMPARE][yuv] "  << path.filename().string()
                      << "  ansB=" << ansB_yuv
                      << "  bpp="  << bpp_yuv
                      << "  Equal=" << (equal_yuv ? "YES" : "NO")
                      << "  Pred=" << pred_ms_yuv << "ms"
                      << "  Rec="  << rec_ms_yuv  << "ms\n";

            const double delta_bpp = bpp_yuv - bpp_rgb; // negative = YUV better
            const double pred_ratio = (double)pred_ms_rgb / std::max(1.0, (double)pred_ms_yuv);
            const double rec_ratio  = (double)rec_ms_rgb  / std::max(1.0, (double)rec_ms_yuv);

            std::cout << "[COMPARE][DELTA] " << path.filename().string()
                      << "  Delta_bpp(yuv-RGB)=" << delta_bpp
                      << "  Pred_RGB/yuv=" << pred_ratio
                      << "  Rec_RGB/yuv="  << rec_ratio  << "\n";

            continue; // do not go to normal single processing
        }

        if (mode == "rgb") {
            auto tPred0 = std::chrono::high_resolution_clock::now();
            auto residuals  = compute_residuals_MED_u8(rgb);
            auto tPred1 = std::chrono::high_resolution_clock::now();

            if (saveVis) {
                auto vis = residuals_visual_rgb8(residuals, rgb);
                save_png(with_suffix_png(path, outDir, "_residuals_vis_rgb").string(), vis);
            }

            auto ansPath = with_suffix_ext(path, outDir, "_rgb", ".r16ans");
            ans::compress_to_file(residuals, /*mode=*/0, rgb.w, rgb.h, rgb.c, ansPath.string());

            auto rec = reconstruct_from_residuals_MED(residuals, rgb);
            auto tRec1 = std::chrono::high_resolution_clock::now();
            save_image(with_suffix_and_same_ext(path, outDir, "_reconstructed").string(), rec);

            st.ans_bytes = file_size_bytes(ansPath.string());
            st.bpp = st.pixels ? (8.0 * (double)st.ans_bytes) / (double)st.pixels : 0.0;
            st.ratio_vs_resid = st.pixels ? ((double)st.ans_bytes / (double)(st.pixels * 2ull)) : 0.0;
            st.ratio_vs_rawrgb = (double)st.ans_bytes /
                                 (double)((uint64_t)rgb.w * rgb.h * 3ull);

            st.t_pred_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tPred1 - tPred0).count();
            st.t_rec_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(tRec1  - tPred1).count();

            double mpix = ((double)st.pixels) / 1e6;
            st.thr_pred_mpps = st.t_pred_ms > 0 ? (1000.0 * mpix / (double)st.t_pred_ms) : 0.0;
            st.thr_rec_mpps  = st.t_rec_ms  > 0 ? (1000.0 * mpix / (double)st.t_rec_ms)  : 0.0;

            st.equal = images_equal(rgb, rec);
            allStats.push_back(st);

            std::cout << "[MODE=RGB] " << st.file << "  Equal: " << (st.equal ? "YES" : "NO") << "\n";

        } else if (mode == "yuv") {
            Image16 yuv = rgb_to_yuv(rgb);

            auto tPred0 = std::chrono::high_resolution_clock::now();
            auto residuals16 = compute_residuals_MED_s16(yuv);
            auto tPred1 = std::chrono::high_resolution_clock::now();

            if (saveVis) {
                auto vis = residuals_visual_s16(residuals16, yuv);
                save_png(with_suffix_png(path, outDir, "_residuals_vis_yuv").string(), vis);
            }

            auto ansPath = with_suffix_ext(path, outDir, "_yuv", ".r16ans");
            ans::compress_to_file(residuals16, /*mode=*/1, yuv.w, yuv.h, yuv.c, ansPath.string());

            auto yuv_rec = reconstruct_from_residuals_MED_s16(residuals16, yuv);
            Image rec = yuv_to_rgb(yuv_rec);
            auto tRec1 = std::chrono::high_resolution_clock::now();
            save_image(with_suffix_and_same_ext(path, outDir, "_reconstructed").string(), rec);

            st.ans_bytes = file_size_bytes(ansPath.string());
            st.bpp = st.pixels ? (8.0 * (double)st.ans_bytes) / (double)st.pixels : 0.0;
            st.ratio_vs_resid = st.pixels ? ((double)st.ans_bytes / (double)(st.pixels * 2ull)) : 0.0;
            st.ratio_vs_rawrgb = (double)st.ans_bytes /
                                 (double)((uint64_t)rgb.w * rgb.h * 3ull);

            st.t_pred_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tPred1 - tPred0).count();
            st.t_rec_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(tRec1  - tPred1).count();

            double mpix = ((double)st.pixels) / 1e6;
            st.thr_pred_mpps = st.t_pred_ms > 0 ? (1000.0 * mpix / (double)st.t_pred_ms) : 0.0;
            st.thr_rec_mpps  = st.t_rec_ms  > 0 ? (1000.0 * mpix / (double)st.t_rec_ms)  : 0.0;

            st.equal = images_equal(rgb, rec);
            allStats.push_back(st);

            std::cout << "[MODE=yuv] " << st.file << "  Equal: " << (st.equal ? "YES" : "NO") << "\n";

        } else if (mode == "ls") {
            if (lsOn == "rgb") {
                auto tPred0 = std::chrono::high_resolution_clock::now();
                auto residuals = compute_residuals_LS_u8(rgb, N, winW, winH);
                auto tPred1 = std::chrono::high_resolution_clock::now();

                st.ls_count  = (long long)g_last_ls_breakdown.used_ls;
                st.med_count = (long long)g_last_ls_breakdown.used_med;
                if (st.ls_count >= 0 && st.med_count >= 0) {
                    auto tot = st.ls_count + st.med_count;
                    st.ls_pct = tot ? (100.0 * (double)st.ls_count / (double)tot) : 0.0;
                }

                if (saveVis) {
                    auto vis = residuals_visual_rgb8(residuals, rgb);
                    save_png(with_suffix_png(path, outDir, "_residuals_vis_ls_rgb").string(), vis);
                }

                auto ansPath = with_suffix_ext(path, outDir, "_ls_rgb", ".r16ans");
                ans::compress_to_file(residuals, /*mode=*/0, rgb.w, rgb.h, rgb.c, ansPath.string());

                auto rec = reconstruct_from_residuals_LS_u8(residuals, rgb, N, winW, winH);
                auto tRec1 = std::chrono::high_resolution_clock::now();
                save_image(with_suffix_and_same_ext(path, outDir, "_reconstructed").string(), rec);

                st.ans_bytes = file_size_bytes(ansPath.string());
                st.bpp = st.pixels ? (8.0 * (double)st.ans_bytes) / (double)st.pixels : 0.0;
                st.ratio_vs_resid = st.pixels ? ((double)st.ans_bytes / (double)(st.pixels * 2ull)) : 0.0;
                st.ratio_vs_rawrgb = (double)st.ans_bytes /
                                     (double)((uint64_t)rgb.w * rgb.h * 3ull);

                st.t_pred_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tPred1 - tPred0).count();
                st.t_rec_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(tRec1  - tPred1).count();

                double mpix = ((double)st.pixels) / 1e6;
                st.thr_pred_mpps = st.t_pred_ms > 0 ? (1000.0 * mpix / (double)st.t_pred_ms) : 0.0;
                st.thr_rec_mpps  = st.t_rec_ms  > 0 ? (1000.0 * mpix / (double)st.t_rec_ms)  : 0.0;

                st.equal = images_equal(rgb, rec);
                allStats.push_back(st);

                std::cout << "Prediction stats: LS=" << g_last_ls_breakdown.used_ls
                          << " MED=" << g_last_ls_breakdown.used_med
                          << " Total=" << (size_t)rgb.w*rgb.h*rgb.c
                          << " (" << (100.0 * (double)g_last_ls_breakdown.used_ls /
                                      (double)((size_t)rgb.w*rgb.h*rgb.c)) << "% LS)\n";
                std::cout << "[MODE=LS on RGB] " << st.file
                          << "  Equal: " << (st.equal ? "YES" : "NO") << "\n";

            } else if (lsOn == "yuv") {
                Image16 yuv = rgb_to_yuv(rgb);

                auto tPred0 = std::chrono::high_resolution_clock::now();
                auto residuals16 = compute_residuals_LS_s16(yuv, N, winW, winH);
                auto tPred1 = std::chrono::high_resolution_clock::now();

                st.ls_count  = (long long)g_last_ls_breakdown.used_ls;
                st.med_count = (long long)g_last_ls_breakdown.used_med;
                if (st.ls_count >= 0 && st.med_count >= 0) {
                    auto tot = st.ls_count + st.med_count;
                    st.ls_pct = tot ? (100.0 * (double)st.ls_count / (double)tot) : 0.0;
                }

                if (saveVis) {
                    auto vis = residuals_visual_s16(residuals16, yuv);
                    save_png(with_suffix_png(path, outDir, "_residuals_vis_ls_yuv").string(), vis);
                }

                auto ansPath = with_suffix_ext(path, outDir, "_ls_yuv", ".r16ans");
                ans::compress_to_file(residuals16, /*mode=*/1, yuv.w, yuv.h, yuv.c, ansPath.string());

                auto yuv_rec = reconstruct_from_residuals_LS_s16(residuals16, yuv, N, winW, winH);
                Image rec = yuv_to_rgb(yuv_rec);
                auto tRec1 = std::chrono::high_resolution_clock::now();
                save_image(with_suffix_and_same_ext(path, outDir, "_reconstructed").string(), rec);

                st.ans_bytes = file_size_bytes(ansPath.string());
                st.bpp = st.pixels ? (8.0 * (double)st.ans_bytes) / (double)st.pixels : 0.0;
                st.ratio_vs_resid = st.pixels ? ((double)st.ans_bytes / (double)(st.pixels * 2ull)) : 0.0;
                st.ratio_vs_rawrgb = (double)st.ans_bytes /
                                     (double)((uint64_t)rgb.w * rgb.h * 3ull);

                st.t_pred_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tPred1 - tPred0).count();
                st.t_rec_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(tRec1  - tPred1).count();

                double mpix = ((double)st.pixels) / 1e6;
                st.thr_pred_mpps = st.t_pred_ms > 0 ? (1000.0 * mpix / (double)st.t_pred_ms) : 0.0;
                st.thr_rec_mpps  = st.t_rec_ms  > 0 ? (1000.0 * mpix / (double)st.t_rec_ms)  : 0.0;

                st.equal = images_equal(rgb, rec);
                allStats.push_back(st);

                std::cout << "Prediction stats: LS=" << g_last_ls_breakdown.used_ls
                          << " MED=" << g_last_ls_breakdown.used_med
                          << " Total=" << (size_t)rgb.w*rgb.h*rgb.c
                          << " (" << (100.0 * (double)g_last_ls_breakdown.used_ls /
                                      (double)((size_t)rgb.w*rgb.h*rgb.c)) << "% LS)\n";
                std::cout << "[MODE=LS on yuv] " << st.file
                          << "  Equal: " << (st.equal ? "YES" : "NO") << "\n";
            } else {
                std::cerr << "Unknown IMG_LS_ON value: " << lsOn << " (use rgb|yuv)\n";
            }
        } else {
            std::cerr << "Unknown IMG_MODE value: " << mode << " (use rgb|yuv|ls)\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error on file \"" << path.string() << "\": " << e.what() << "\n";
    }
}
    write_batch_summary(outDir, allStats);
    return 0;

} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
}
}