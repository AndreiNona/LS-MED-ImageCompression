#include "imageIO.h"
#include "predictor.h"

#include <gtest/gtest.h>
#include <chrono>
#include <cstdlib>     // std::getenv
#include <string>
#include <utility>

using clock_hr = std::chrono::high_resolution_clock;

enum class Pipeline {
    MED_RGB,
    MED_RCT,
    LS_RGB,
    LS_RCT
};

struct ModeCfg {
    Pipeline pipe;
    int N;
    int winW;
    int winH;
    const char* name;
};

// ---------- helpers ----------

static std::pair<long long, long long> run_roundtrip(
    const ModeCfg& cfg,
    const Image& srcRGB,
    Image& outRec
) {
    long long pred_ms = 0;
    long long recon_ms = 0;

    if (cfg.pipe == Pipeline::MED_RGB) {
        auto t1 = clock_hr::now();
        auto residuals = compute_residuals_MED_u8(srcRGB);
        auto t2 = clock_hr::now();

        outRec = reconstruct_from_residuals_MED(residuals, srcRGB);
        auto t3 = clock_hr::now();

        pred_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        recon_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        return {pred_ms, recon_ms};
    }

    if (cfg.pipe == Pipeline::MED_RCT) {
        Image16 rct = rct_from_rgb(srcRGB);

        auto t1 = clock_hr::now();
        auto residuals16 = compute_residuals_MED_s16(rct);
        auto t2 = clock_hr::now();

        Image16 rct_rec = reconstruct_from_residuals_MED_s16(residuals16, rct);
        outRec = rct_to_rgb(rct_rec);
        auto t3 = clock_hr::now();

        pred_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        recon_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        return {pred_ms, recon_ms};
    }

    if (cfg.pipe == Pipeline::LS_RGB) {
        auto t1 = clock_hr::now();
        auto residuals = compute_residuals_LS_u8(srcRGB, cfg.N, cfg.winW, cfg.winH);
        auto t2 = clock_hr::now();

        outRec = reconstruct_from_residuals_LS_u8(residuals, srcRGB, cfg.N, cfg.winW, cfg.winH);
        auto t3 = clock_hr::now();

        pred_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        recon_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        return {pred_ms, recon_ms};
    }

    // LS_RCT
    Image16 rct = rct_from_rgb(srcRGB);

    auto t1 = clock_hr::now();
    auto residuals16 = compute_residuals_LS_s16(rct, cfg.N, cfg.winW, cfg.winH);
    auto t2 = clock_hr::now();

    Image16 rct_rec = reconstruct_from_residuals_LS_s16(residuals16, rct, cfg.N, cfg.winW, cfg.winH);
    outRec = rct_to_rgb(rct_rec);
    auto t3 = clock_hr::now();

    pred_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    recon_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    return {pred_ms, recon_ms};
}



class RoundTripParamTest : public ::testing::TestWithParam<ModeCfg> {};

TEST_P(RoundTripParamTest, BitExactAndTiming) {

    const char* p = std::getenv("TEST_IMAGE");
    if (!p) GTEST_SKIP() << "Set TEST_IMAGE env var to an existing PNG/JPG";

    Image src;
    try {
        src = load_image(p);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Failed to load TEST_IMAGE: " << e.what();
    }

    const ModeCfg cfg = GetParam();

    Image rec;
    auto [pred_ms, recon_ms] = run_roundtrip(cfg, src, rec);

    ASSERT_EQ(src.w, rec.w);
    ASSERT_EQ(src.h, rec.h);
    ASSERT_EQ(src.c, rec.c);
    ASSERT_EQ(src.px, rec.px) << "Reconstructed image differs from source";

    // Timings
    const double pixels = double(src.w) * src.h * src.c;
    const double pred_ms_per_mpix  = pred_ms  / (pixels / 1e6);
    const double recon_ms_per_mpix = recon_ms / (pixels / 1e6);

    std::cout << "[TEST] " << cfg.name
              << "  " << src.w << "x" << src.h << "x" << src.c
              << "  Predict: " << pred_ms << " ms (" << pred_ms_per_mpix << " ms/Mpix)"
              << "  Reconstruct: " << recon_ms << " ms (" << recon_ms_per_mpix << " ms/Mpix)\n";

    RecordProperty(std::string("Predict_ms_") + cfg.name, pred_ms);
    RecordProperty(std::string("Reconstruct_ms_") + cfg.name, recon_ms);
}


static const ModeCfg kModes[] = {
    { Pipeline::MED_RGB, 0,0,0, "MED_RGB" },
    { Pipeline::MED_RCT, 0,0,0, "MED_RCT" },
    { Pipeline::LS_RGB,  3,3,3, "LS_RGB_N3_W3x3" },
    { Pipeline::LS_RCT,  3,3,3, "LS_RCT_N3_W3x3" },
};

INSTANTIATE_TEST_SUITE_P(AllPipelines,
                         RoundTripParamTest,
                         ::testing::ValuesIn(kModes));
