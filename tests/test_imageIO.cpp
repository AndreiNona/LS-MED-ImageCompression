//
// Created by ioana on 22-Sep-25.
//

// tests/imageio_tests.cpp
#include "imageIO.h"
#include <gtest/gtest.h>

#include <random>
#include <tuple>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iostream>

using clock_hr = std::chrono::high_resolution_clock;

// ---------- helpers  ----------
namespace {

Image make_const_image(int w, int h, unsigned char r, unsigned char g, unsigned char b) {
    Image im; im.w=w; im.h=h; im.c=3; im.px.resize(static_cast<size_t>(w)*h*3);
    for (int i=0;i<w*h;++i) { im.px[3*i+0]=r; im.px[3*i+1]=g; im.px[3*i+2]=b; }
    return im;
}

Image make_gradient(int w, int h) {
    Image im; im.w=w; im.h=h; im.c=3; im.px.resize(static_cast<size_t>(w)*h*3);
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        unsigned char r = static_cast<unsigned char>((w>1) ? (x*255)/(w-1) : 0);
        unsigned char g = static_cast<unsigned char>((h>1) ? (y*255)/(h-1) : 0);
        unsigned char b = static_cast<unsigned char>(((w+h>2) ? (x+y)*255/(w+h-2) : 0));
        const int i = y*w + x;
        im.px[3*i+0]=r; im.px[3*i+1]=g; im.px[3*i+2]=b;
    }
    return im;
}

Image make_random(int w, int h, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(0,255);
    Image im; im.w=w; im.h=h; im.c=3; im.px.resize(static_cast<size_t>(w)*h*3);
    for (int i=0;i<w*h;++i) {
        im.px[3*i+0]=static_cast<unsigned char>(d(rng));
        im.px[3*i+1]=static_cast<unsigned char>(d(rng));
        im.px[3*i+2]=static_cast<unsigned char>(d(rng));
    }
    return im;
}

}
// namespace

// ---------- Tests: YUV  round trip ----------

TEST(ImageIO_YUV, RoundTrip_ConstColors) {
    std::vector<std::tuple<unsigned char,unsigned char,unsigned char>> cases = {
        {0,0,0}, {255,255,255}, {255,0,0}, {0,255,0}, {0,0,255},
        {1,2,3}, {254,1,128}, {17,200,33}
    };
    for (auto [r,g,b] : cases) {
        Image src = make_const_image(23,17,r,g,b);
        Image16 yuv = rgb_to_yuv(src);
        Image rec = yuv_to_rgb(yuv);
        EXPECT_TRUE(images_equal(src, rec)) << "Const color round trip failed";
    }
}

TEST(ImageIO_YUV, RoundTrip_Gradients_VariousSizes) {
    for (auto wh : { std::pair{1,1}, std::pair{2,2}, std::pair{3,5},
                     std::pair{16,16}, std::pair{63,47}, std::pair{256,129} }) {
        Image src = make_gradient(wh.first, wh.second);
        Image16 yuv = rgb_to_yuv(src);
        Image rec = yuv_to_rgb(yuv);
        EXPECT_TRUE(images_equal(src, rec)) << "Gradient round trip failed for "
                                            << wh.first << "x" << wh.second;
    }
}

TEST(ImageIO_YUV, RoundTrip_Random_Fuzz) {
    int seeds[] = {0,1,2,12345,987654321};
    for (int s : seeds) {
        for (auto wh : { std::pair{7,7}, std::pair{31,9}, std::pair{64,64}, std::pair{127,63} }) {
            Image src = make_random(wh.first, wh.second, static_cast<uint32_t>(s));
            Image16 yuv = rgb_to_yuv(src);
            Image rec = yuv_to_rgb(yuv);
            EXPECT_TRUE(images_equal(src, rec)) << "Random round trip failed (seed=" << s
                                                << ", " << wh.first << "x" << wh.second << ")";
        }
    }
}

TEST(ImageIO_YUV, UV_In_Range) {
    std::vector<std::tuple<unsigned char,unsigned char,unsigned char>> extremes = {
        {255,0,0},   // V=+255
        {0,255,0},   // U=V=-255
        {0,0,255},   // U=+255
        {255,255,0}, // U=-255
        {0,255,255}, // V=-255
        {255,0,255}, // U=+255, V=+255
    };
    for (auto [r,g,b] : extremes) {
        Image src = make_const_image(5,3,r,g,b);
        Image16 yuv = rgb_to_yuv(src);
        for (int i=0;i<5*3;++i) {
            int16_t U = yuv.px[3*i+1];
            int16_t V = yuv.px[3*i+2];
            EXPECT_GE(U, -255); EXPECT_LE(U, 255);
            EXPECT_GE(V, -255); EXPECT_LE(V, 255);
        }
        Image rec = yuv_to_rgb(yuv);
        EXPECT_TRUE(images_equal(src, rec));
    }
}

// ---------- Tests:  original RCT pair (not needed) ----------

TEST(ImageIO_RCT, RoundTrip_Random) {
    for (auto wh : { std::pair{8,8}, std::pair{17,13}, std::pair{64,31} }) {
        Image src = make_random(wh.first, wh.second, 424242u);
        Image16 rct = rct_from_rgb(src);
        Image rec = rct_to_rgb(rct);
        EXPECT_TRUE(images_equal(src, rec)) << "RCT round trip failed";
    }
}

TEST(ImageIO_IO, LoadAndRoundTripIfEnvSet) {
    const char* p = std::getenv("TEST_IMAGE");
    if (!p) GTEST_SKIP() << "Set TEST_IMAGE to a PNG/JPG to enable this test";
    Image src;
    ASSERT_NO_THROW(src = load_image(p));
    ASSERT_EQ(src.c, 3) << "load_image drops alpha; test expects RGB image";

    // YUV (lossless/RCT-style)
    {
        auto t1 = clock_hr::now();
        Image16 yuv = rgb_to_yuv(src);
        auto t2 = clock_hr::now();
        Image rec = yuv_to_rgb(yuv);
        auto t3 = clock_hr::now();

        EXPECT_TRUE(images_equal(src, rec));
        const auto enc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        const auto dec_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        std::cout << "[YUV] encode=" << enc_ms << "ms  decode=" << dec_ms << "ms\n";
        RecordProperty("YUV_encode_ms", enc_ms);
        RecordProperty("YUV_decode_ms", dec_ms);
    }
    {
        auto t1 = clock_hr::now();
        Image16 rct = rct_from_rgb(src);
        auto t2 = clock_hr::now();
        Image rec = rct_to_rgb(rct);
        auto t3 = clock_hr::now();

        EXPECT_TRUE(images_equal(src, rec));
        const auto enc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        const auto dec_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        std::cout << "[RCT] encode=" << enc_ms << "ms  decode=" << dec_ms << "ms\n";
        RecordProperty("RCT_encode_ms", enc_ms);
        RecordProperty("RCT_decode_ms", dec_ms);
    }
}
