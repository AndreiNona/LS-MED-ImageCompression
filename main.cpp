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
        std::string mode = "rgb"; // rgb | rct | ls
        std::string lsOn = "rgb"; // for mode=ls: run LS on rgb (default) or rct
        std::string saveResPath;  // --save-res=...
        std::string loadResPath;  // --load-res=...
        bool saveVis = false;     // --save-res-vis
        int  N = 4, winW = 4, winH = 4; // LS params

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--mode=",0) == 0)        mode = a.substr(7);
            else if (a.rfind("--ls-on=",0)==0)    lsOn = a.substr(8); // rgb|rct
            else if (a.rfind("--ls-N=",0)==0)     N = std::max(1, std::stoi(a.substr(7)));
            else if (a.rfind("--ls-win=",0)==0) {
                auto s = a.substr(9);
                auto pos = s.find('x');
                if (pos!=std::string::npos) { winW = std::stoi(s.substr(0,pos)); winH = std::stoi(s.substr(pos+1)); }
            }
            else if (a.rfind("--save-res=",0)==0) saveResPath = a.substr(11);
            else if (a.rfind("--load-res=",0)==0) loadResPath = a.substr(11);
            else if (a == "--save-res-vis")       saveVis = true;
        }

        // ---- reconstruct from residual file (skip prediction) ----
        if (!loadResPath.empty()) {
            auto rf = load_residuals(loadResPath);
            if (rf.mode == 0) {
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
                Image16 shape; shape.w = rf.w; shape.h = rf.h; shape.c = rf.c;
                auto t0 = std::chrono::high_resolution_clock::now();
                Image16 rct_rec = reconstruct_from_residuals_MED_s16(rf.residuals, shape);
                Image rec = rct_to_rgb(rct_rec);
                auto t1 = std::chrono::high_resolution_clock::now();
                save_png("out_reconstructed_from_file.png", rec);
                std::cout << "[FROM FILE] mode=RCT  "
                          << rf.w << "x" << rf.h << "x" << rf.c
                          << " | Reconstruct: "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                          << " ms\n";
            }
            return 0;
        }

        // ---- forward path ----
        auto tLoad0 = std::chrono::high_resolution_clock::now();
        Image rgb = load_image(in);
        auto tLoad1 = std::chrono::high_resolution_clock::now();

        if (mode == "rgb") {
            auto tPred0 = std::chrono::high_resolution_clock::now();
            auto residuals  = compute_residuals_MED_u8(rgb);
            auto tPred1 = std::chrono::high_resolution_clock::now();

            if (saveVis) save_png("out_residuals_vis.png", residuals_visual_rgb8(residuals, rgb));
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
        else if (mode == "rct") {
            Image16 rct = rct_from_rgb(rgb);

            auto tPred0 = std::chrono::high_resolution_clock::now();
            auto residuals16 = compute_residuals_MED_s16(rct);
            auto tPred1 = std::chrono::high_resolution_clock::now();

            if (saveVis) {
                auto vis_rct = residuals_visual_s16(residuals16, rct);
                save_png("out_residuals_vis_rct.png", vis_rct);
            }
            if (!saveResPath.empty()) save_residuals(saveResPath, /*mode=*/1, rct.w, rct.h, rct.c, residuals16);

            auto rct_rec = reconstruct_from_residuals_MED_s16(residuals16, rct);
            Image rec = rct_to_rgb(rct_rec);
            auto tRec1 = std::chrono::high_resolution_clock::now();

            save_png("out_reconstructed.png", rec);

            std::cout << "[MODE=RCT] Equal: " << (images_equal(rgb, rec) ? "YES" : "NO") << "\n";
            std::cout << "I/O: " << std::chrono::duration_cast<std::chrono::milliseconds>(tLoad1-tLoad0).count()
                      << " ms | Predict: " << std::chrono::duration_cast<std::chrono::milliseconds>(tPred1-tPred0).count()
                      << " ms | Reconstruct: " << std::chrono::duration_cast<std::chrono::milliseconds>(tRec1-tPred1).count()
                      << " ms\n";
        }
        else if (mode == "ls") {
            if (lsOn == "rgb") {
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
            } else { // lsOn == "rct"
                Image16 rct = rct_from_rgb(rgb);

                auto tPred0 = std::chrono::high_resolution_clock::now();
                auto residuals16 = compute_residuals_LS_s16(rct, N, winW, winH);
                auto tPred1 = std::chrono::high_resolution_clock::now();

                if (saveVis) {
                    auto vis_rct = residuals_visual_s16(residuals16, rct);
                    save_png("out_residuals_vis_ls_rct.png", vis_rct);
                }
                if (!saveResPath.empty()) save_residuals(saveResPath, /*mode=*/1, rct.w, rct.h, rct.c, residuals16);

                auto rct_rec = reconstruct_from_residuals_LS_s16(residuals16, rct, N, winW, winH);
                Image rec = rct_to_rgb(rct_rec);
                auto tRec1 = std::chrono::high_resolution_clock::now();

                save_png("out_reconstructed.png", rec);

                std::cout << "[MODE=LS on RCT] Equal: " << (images_equal(rgb, rec) ? "YES" : "NO") << "\n";
                std::cout << "I/O: " << std::chrono::duration_cast<std::chrono::milliseconds>(tLoad1-tLoad0).count()
                          << " ms | Predict: " << std::chrono::duration_cast<std::chrono::milliseconds>(tPred1-tPred0).count()
                          << " ms | Reconstruct: " << std::chrono::duration_cast<std::chrono::milliseconds>(tRec1-tPred1).count()
                          << " ms\n";
            }
        }
        else {
            std::cerr << "Unknown mode: " << mode << "  (use --mode=rgb | --mode=rct | --mode=ls)\n";
            return 2;
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
