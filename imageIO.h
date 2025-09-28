#pragma once
#include <string>
#include <vector>
#include <cstdint>


enum class ImageFormat { Unknown, PNG, JPG, BMP, TGA, PPM, PGM };

// 8-bit
struct Image {
    int w = 0, h = 0, c = 0;
    std::vector<unsigned char> px;
    ImageFormat format = ImageFormat::Unknown;
};

// 16-bit
struct Image16 {
    int w = 0, h = 0, c = 0; // expect c==3
    std::vector<int16_t> px;
};

// -------- I/O  --------
Image load_image(const std::string& path);          // throws on error
void   save_png  (const std::string& path, const Image& im); // throws on error
void   save_image(const std::string& path, const Image& im);

// -------- reversible --------
Image16 rgb_to_yuv(const Image& rgb);
Image yuv_to_rgb(const Image16& yuv);
// -------- Not Used --------
Image16 rct_from_rgb(const Image& rgb);
Image   rct_to_rgb  (const Image16& rct);


bool images_equal(const Image& a, const Image& b);
