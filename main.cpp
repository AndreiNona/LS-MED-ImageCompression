#include "imageIO.h"
#include "predictor.h"
#include "residualIO.h"

#include <iostream>
#include <chrono>
#include <string>

int main(int argc, char** argv) {
    try {
        // ---- args ----
        std::string in  = (argc >= 2 && std::string(argv[1]).rfind("--",0)!=0)
                        ? argv[1] : "test_images/test.png"; // first non-flag arg as input

        // Modes:
        //   --mode=rgb   : MED on RGB/u8
        //   --mode=yuvr  : MED on reversible YUV (your integer transform, s16)
        //   --mode=ls    : LS predictor on either RGB or YUVR, chosen via --ls-on
        // Back-compat aliases: "--mode=rct" behaves like "--mode=yuvr"
        std::string mode = "rgb";   // rgb | yuvr | ls   (alias: rct -> yuvr)

        // For LS:
        //   --ls-on=rgb
        //   --ls-on=yuvr  (alias: rct -> yuvr)
        std::string lsOn = "rgb";

        std::string saveResPath;    // --save-res=...
        std::string loadResPath;    // --load-res=...
        bool saveVis = false;       // --save-res-vis
        int  N = 4, winW = 4, winH = 4; // LS params

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--mode=",0) == 0)        mode = a.substr(7);
            else if (a.rfind("--ls-on=",0)==0)    lsOn = a.substr(8);
            else if (a.rfind("--ls-N=",0)==0)     N = std::max(1, std::stoi(a.substr(7)));
            else if (a.rfind("--ls-win=",0)==0) {
                auto s = a.substr(9);
                auto pos = s.find('x');
                if (pos!=std::string::npos) {
                    winW = std::stoi(s.substr(0,pos));
                    winH = std::stoi(s.substr(pos+1));
                }
            }
            else if (a.rfind("--save-res=",0)==0) saveResPath = a.substr(11);
            else if (a.rfind("--load-res=",0)==0) loadResPath = a.substr(11);
            else if (a == "--save-res-vis")       saveVis = true;
        }

        // Back-compatibility
        auto normalize_space = [](std::string s){
            for (auto& ch : s) ch = (char)std::tolower((unsigned char)ch);
            if (s == "rct") s = "yuvr";
            return s;
        };
        mode = normalize_space(mode);
        lsOn = normalize_space(lsOn);

        // ---- reconstruct from residual file (skip prediction) ----
        if (!loadResPath.empty()) {
            auto rf = load_residuals(loadResPath);
            if (rf.mode == 0) {
                // uint8/RGB residuals (MED on RGB)
                Image shape; shape.w = rf.w; shape.h = rf.h; shape.c = rf.c;
                auto t0 = std::chrono::high_resolution_clock::now();
                Image rec = reconstruct_from_residuals_MED(rf.residuals, shape);
                auto t1 = std::chrono::high_resolution_clock::now();
                save_png("out_reconstructed_from_file.png", rec);
                std::cout << "[FROM FILE] mode=RGB  "
                          << rf.w << "x" << rf.h << "x" << rf.c
                          << " | Reconstruct: "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                          << " ms\n";
            } else {
                // int16
                Image16 shape; shape.w = rf.w; shape.h = rf.h; shape.c = rf.c;
                auto t0 = std::chrono::high_resolution_clock::now();
                Image16 yuvr_rec = reconstruct_from_residuals_MED_s16(rf.residuals, shape);
                Image rec = yuv_to_rgb(yuvr_rec);
                auto t1 = std::chrono::high_resolution_clock::now();
                save_png("out_reconstructed_from_file.png", rec);
                std::cout << "[FROM FILE] mode=YUVR  "
                          << rf.w << "x" << rf.h << "x" << rf.c
                          << " | Reconstruct: "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                          << " ms\n";
            }
            return 0;
        }

        auto tLoad0 = std::chrono::high_resolution_clock::now();
        Image rgb = load_image(in);
        auto tLoad1 = std::chrono::high_resolution_clock::now();

        if (mode == "rgb") {
            // MED on RGB/u8
            auto tPred0 = std::chrono::high_resolution_clock::now();
            auto residuals  = compute_residuals_MED_u8(rgb);
            auto tPred1 = std::chrono::high_resolution_clock::now();

            if (saveVis) save_png("out_residuals_vis_rgb.png", residuals_visual_rgb8(residuals, rgb));
            if (!saveResPath.empty()) save_residuals(saveResPath, /*mode=*/0, rgb.w, rgb.h, rgb.c, residuals);

            auto rec = reconstruct_from_residuals_MED(residuals, rgb);
            auto tRec1 = std::chrono::high_resolution_clock::now();

            save_png("out_reconstructed.png", rec);

            std::cout << "[MODE=RGB] Equal: " << (images_equal(rgb, rec) ? "YES" : "NO") << "\n";
            std::cout << "I/O: " << std::chrono::duration_cast<std::chrono::milliseconds>(tLoad1-tLoad0).count()
                      << " ms | Predict: " << std::chrono::duration_cast<std::chrono::milliseconds>(tPred1-tPred0).count()
                      << " ms | Reconstruct: " << std::chrono::duration_cast<std::chrono::milliseconds>(tRec1-tPred1).count()
                      << " ms\n";
        }
        else if (mode == "yuvr") {
            // MED on reversible YUV (int16 domain)
            Image16 yuvr = rgb_to_yuv(rgb);

            auto tPred0 = std::chrono::high_resolution_clock::now();
            auto residuals16 = compute_residuals_MED_s16(yuvr);
            auto tPred1 = std::chrono::high_resolution_clock::now();

            if (saveVis) {
                auto vis = residuals_visual_s16(residuals16, yuvr);
                save_png("out_residuals_vis_yuvr.png", vis);
            }
            if (!saveResPath.empty()) save_residuals(saveResPath, /*mode=*/1, yuvr.w, yuvr.h, yuvr.c, residuals16);

            auto yuvr_rec = reconstruct_from_residuals_MED_s16(residuals16, yuvr);
            Image rec = yuv_to_rgb(yuvr_rec);
            auto tRec1 = std::chrono::high_resolution_clock::now();

            save_png("out_reconstructed.png", rec);

            std::cout << "[MODE=YUVR] Equal: " << (images_equal(rgb, rec) ? "YES" : "NO") << "\n";
            std::cout << "I/O: " << std::chrono::duration_cast<std::chrono::milliseconds>(tLoad1-tLoad0).count()
                      << " ms | Predict: " << std::chrono::duration_cast<std::chrono::milliseconds>(tPred1-tPred0).count()
                      << " ms | Reconstruct: " << std::chrono::duration_cast<std::chrono::milliseconds>(tRec1-tPred1).count()
                      << " ms\n";
        }
        else if (mode == "ls") {
            if (lsOn == "rgb") {
                // LS on RGB/u8
                auto tPred0 = std::chrono::high_resolution_clock::now();
                auto residuals = compute_residuals_LS_u8(rgb, N, winW, winH);
                auto tPred1 = std::chrono::high_resolution_clock::now();

                if (saveVis) save_png("out_residuals_vis_ls_rgb.png", residuals_visual_rgb8(residuals, rgb));
                if (!saveResPath.empty()) save_residuals(saveResPath, /*mode=*/0, rgb.w, rgb.h, rgb.c, residuals);

                auto rec = reconstruct_from_residuals_LS_u8(residuals, rgb, N, winW, winH);
                auto tRec1 = std::chrono::high_resolution_clock::now();

                save_png("out_reconstructed.png", rec);

                std::cout << "[MODE=LS on RGB] Equal: " << (images_equal(rgb, rec) ? "YES" : "NO") << "\n";
                std::cout << "I/O: " << std::chrono::duration_cast<std::chrono::milliseconds>(tLoad1-tLoad0).count()
                          << " ms | Predict: " << std::chrono::duration_cast<std::chrono::milliseconds>(tPred1-tPred0).count()
                          << " ms | Reconstruct: " << std::chrono::duration_cast<std::chrono::milliseconds>(tRec1-tPred1).count()
                          << " ms\n";
            } else if (lsOn == "yuvr") {
                // LS on reversible YUV (int16)
                Image16 yuvr = rgb_to_yuv(rgb);

                auto tPred0 = std::chrono::high_resolution_clock::now();
                auto residuals16 = compute_residuals_LS_s16(yuvr, N, winW, winH);
                auto tPred1 = std::chrono::high_resolution_clock::now();

                if (saveVis) {
                    auto vis = residuals_visual_s16(residuals16, yuvr);
                    save_png("out_residuals_vis_ls_yuvr.png", vis);
                }
                if (!saveResPath.empty()) save_residuals(saveResPath, /*mode=*/1, yuvr.w, yuvr.h, yuvr.c, residuals16);

                auto yuvr_rec = reconstruct_from_residuals_LS_s16(residuals16, yuvr, N, winW, winH);
                Image rec = yuv_to_rgb(yuvr_rec);
                auto tRec1 = std::chrono::high_resolution_clock::now();

                save_png("out_reconstructed.png", rec);

                std::cout << "[MODE=LS on YUVR] Equal: " << (images_equal(rgb, rec) ? "YES" : "NO") << "\n";
                std::cout << "I/O: " << std::chrono::duration_cast<std::chrono::milliseconds>(tLoad1-tLoad0).count()
                          << " ms | Predict: " << std::chrono::duration_cast<std::chrono::milliseconds>(tPred1-tPred0).count()
                          << " ms | Reconstruct: " << std::chrono::duration_cast<std::chrono::milliseconds>(tRec1-tPred1).count()
                          << " ms\n";
            } else {
                std::cerr << "Unknown --ls-on value: " << lsOn << "  (use rgb | yuvr)\n";
                return 2;
            }
        }
        else {
            std::cerr << "Unknown --mode value: " << mode << "  (use rgb | yuvr | ls)\n";
            return 2;
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}