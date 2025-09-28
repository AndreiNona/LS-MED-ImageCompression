#include "imageIO.h"
#include <stdexcept>
#include <algorithm>
#include <fstream>


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <cctype>

#include "stb_image.h"
#include "stb_image_write.h"

//------- Prevent overflow -------
static inline unsigned char clamp8(int v) {
    return (unsigned char)std::clamp(v, 0, 255);
}
static inline unsigned char clamp8i(int v) {
    return clamp8(v);
}
static inline int floor_div4(int x) {
    return (x >= 0) ? (x >> 2) : -(((-x) + 3) >> 2);
}

static ImageFormat detect_format_from_path(const std::string& path) {
    auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return ImageFormat::Unknown;
    std::string ext = path.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char ch){ return (char)std::tolower(ch); });

    if (ext == "png") return ImageFormat::PNG;
    if (ext == "jpg" || ext == "jpeg") return ImageFormat::JPG;
    if (ext == "bmp") return ImageFormat::BMP;
    if (ext == "tga") return ImageFormat::TGA;
    if (ext == "ppm") return ImageFormat::PPM;
    if (ext == "pgm") return ImageFormat::PGM;
    if (ext == "pnm") return ImageFormat::PPM;
    return ImageFormat::Unknown;
}

static void ensure_channels_for_target(Image& im, ImageFormat fmt) {
    auto requires_rgb =
        (fmt == ImageFormat::JPG) || // stb_jpg writer expects 3 components
        (fmt == ImageFormat::BMP) || // stb_bmp writer expects 3 or 4; keep 3
        (fmt == ImageFormat::TGA);   // stb_tga writer expects 3 or 4; keep 3

    if (requires_rgb && im.c == 1) {
        std::vector<unsigned char> rgb;
        rgb.resize((size_t)im.w * im.h * 3);
        for (int i = 0; i < im.w * im.h; ++i) {
            unsigned char g = im.px[i];
            rgb[3*i+0] = g;
            rgb[3*i+1] = g;
            rgb[3*i+2] = g;
        }
        im.px.swap(rgb);
        im.c = 3;
    }
}
//------- load image here -------
Image load_image(const std::string& path) {
    Image im;
    int w, h, c;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 0);
    if (!data) throw std::runtime_error("Failed to load image: " + path);

    im.w = w; im.h = h; im.c = c; im.format = detect_format_from_path(path);
    im.px.assign(data, data + static_cast<size_t>(w) * h * c);
    stbi_image_free(data);

    // Drop alpha to RGB
    if (im.c == 4) {
        Image tmp; tmp.w = im.w; tmp.h = im.h; tmp.c = 3; tmp.format = im.format;
        tmp.px.resize(static_cast<size_t>(im.w) * im.h * 3);
        for (int i = 0; i < im.w * im.h; ++i) {
            tmp.px[3*i+0] = im.px[4*i+0];
            tmp.px[3*i+1] = im.px[4*i+1];
            tmp.px[3*i+2] = im.px[4*i+2];
        }
        im = std::move(tmp);
    }

    // Enforce supported channel counts
    if (im.c != 1 && im.c != 3)
        throw std::runtime_error("Only Gray or RGB images are supported after load");

    return im;
}

//Not needed, created to enforce save method due to file size discrepancy between original and reconstruction
static void ensure_gray_or_rgb(Image& im) {
    if (im.c == 1 || im.c == 3) return;
    if (im.c == 4) {
        //Alpha to RGB
        Image tmp; tmp.w = im.w; tmp.h = im.h; tmp.c = 3;
        tmp.px.resize((size_t)tmp.w * tmp.h * 3);
        for (int i=0; i<im.w*im.h; ++i) {
            tmp.px[3*i+0] = im.px[4*i+0];
            tmp.px[3*i+1] = im.px[4*i+1];
            tmp.px[3*i+2] = im.px[4*i+2];
        }
        im = std::move(tmp);
        return;
    }
    throw std::runtime_error("save_png: unsupported channel count (must be 1 or 3)");
}

//------- save image in the same format as loaded -------
void save_image(const std::string& path, const Image& im_in) {
    Image im = im_in;
    ensure_gray_or_rgb(im);

    ImageFormat fmt = im.format;
    if (fmt == ImageFormat::Unknown) {
        fmt = detect_format_from_path(path);
        if (fmt == ImageFormat::Unknown) {
            throw std::runtime_error("Unsupported or unknown format for saving: " + path);
        }
    }

    // If target requires RGB
    ensure_channels_for_target(im, fmt);

    const int stride = im.w * im.c;

    switch (fmt) {
    case ImageFormat::PNG:
        if (!stbi_write_png(path.c_str(), im.w, im.h, im.c, im.px.data(), stride))
            throw std::runtime_error("Failed to write PNG: " + path);
        break;
    case ImageFormat::JPG:
        if (!stbi_write_jpg(path.c_str(), im.w, im.h, im.c, im.px.data(), 95))
            throw std::runtime_error("Failed to write JPG: " + path);
        break;
    case ImageFormat::BMP:
        if (!stbi_write_bmp(path.c_str(), im.w, im.h, im.c, im.px.data()))
            throw std::runtime_error("Failed to write BMP: " + path);
        break;
    case ImageFormat::TGA:
        if (!stbi_write_tga(path.c_str(), im.w, im.h, im.c, im.px.data()))
            throw std::runtime_error("Failed to write TGA: " + path);
        break;
    case ImageFormat::PPM:
    case ImageFormat::PGM: {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs) throw std::runtime_error("Failed to open output PPM/PGM: " + path);
        if (im.c == 1) {
            ofs << "P5\n" << im.w << " " << im.h << "\n255\n";
            ofs.write(reinterpret_cast<const char*>(im.px.data()),
                      static_cast<std::streamsize>(im.px.size()));
        } else if (im.c == 3) {
            ofs << "P6\n" << im.w << " " << im.h << "\n255\n";
            ofs.write(reinterpret_cast<const char*>(im.px.data()),
                      static_cast<std::streamsize>(im.px.size()));
        } else {
            throw std::runtime_error("PPM/PGM must be 1 or 3 channels");
        }
        if (!ofs.good())
            throw std::runtime_error("Error writing PPM/PGM: " + path);
        break;
    }
    default:
        throw std::runtime_error("Unsupported or unknown format for saving: " + path);
    }

    std::cout << "Saving " << path << " (" << im.w << "x" << im.h << ", ch=" << im.c << ")\n";
}

//------- save image  -------
void save_png(const std::string& path, const Image& im_in) {
    Image im = im_in;            // make a local copy
    ensure_gray_or_rgb(im);      // enforce 1 or 3 channels
    const int stride = im.w * im.c;
    if (!stbi_write_png(path.c_str(), im.w, im.h, im.c, im.px.data(), stride))
        throw std::runtime_error("Failed to write PNG: " + path);
    std::cout << "Saving " << path << " (" << im.w << "x" << im.h << ", ch=" << im.c << ")\n";
}


// ----------------- Reversible YUV -----------------
Image16 rgb_to_yuv(const Image& rgb) {
    if (rgb.c != 1 && rgb.c != 3)
        throw std::runtime_error("rgb_to_yuv expects Gray(1) or RGB(3)");

    // --- Grayscale: no op (carry Y only) ---
    if (rgb.c == 1) {
        Image16 y_only; y_only.w = rgb.w; y_only.h = rgb.h; y_only.c = 1;
        y_only.px.resize(static_cast<size_t>(rgb.w) * rgb.h);
        for (int i = 0; i < rgb.w * rgb.h; ++i) {
            y_only.px[i] = static_cast<int16_t>(rgb.px[i]); // 0..255 in low 8 bits
        }
        return y_only;
    }

    // --- RGB path ---
    Image16 yuv; yuv.w = rgb.w; yuv.h = rgb.h; yuv.c = 3;
    yuv.px.resize((size_t)yuv.w * yuv.h * 3);

    for (int i = 0; i < rgb.w * rgb.h; ++i) {
        int R = rgb.px[3*i+0];
        int G = rgb.px[3*i+1];
        int B = rgb.px[3*i+2];

        uint8_t Y = (uint8_t)((R + 2*G + B) >> 2);
        int16_t U = (int16_t)(B - G);
        int16_t V = (int16_t)(R - G);

        yuv.px[3*i+0] = (int16_t)Y;
        yuv.px[3*i+1] = U;
        yuv.px[3*i+2] = V;
    }
    return yuv;
}

Image yuv_to_rgb(const Image16& yuv) {
    if (yuv.c != 1 && yuv.c != 3)
        throw std::runtime_error("yuv_to_rgb expects 1 (Gray) or 3 channels");

    // --- Grayscale: no op back to grayscale ---
    if (yuv.c == 1) {
        Image gray; gray.w = yuv.w; gray.h = yuv.h; gray.c = 1;
        gray.px.resize(static_cast<size_t>(gray.w) * gray.h);
        for (int i = 0; i < gray.w * gray.h; ++i) {
            gray.px[i] = clamp8i((int)(uint8_t)yuv.px[i]);
        }
        return gray;
    }

    // --- RGB path  ---
    Image rgb; rgb.w = yuv.w; rgb.h = yuv.h; rgb.c = 3;
    rgb.px.resize((size_t)rgb.w * rgb.h * 3);

    auto floor_div4 = [](int x){ return x >= 0 ? (x >> 2) : -(( -x + 3 ) >> 2); };

    for (int i = 0; i < yuv.w * yuv.h; ++i) {
        int Y = (uint8_t)yuv.px[3*i+0];
        int U = (int)yuv.px[3*i+1];
        int V = (int)yuv.px[3*i+2];

        int G = Y - floor_div4(U + V);
        int R = G + V;
        int B = G + U;

        rgb.px[3*i+0] = clamp8i(R);
        rgb.px[3*i+1] = clamp8i(G);
        rgb.px[3*i+2] = clamp8i(B);
    }
    return rgb;
}

// -------- reversible --------
Image16 rct_from_rgb(const Image& rgb) {
    if (rgb.c != 3) throw std::runtime_error("rct_from_rgb expects RGB");
    Image16 rct; rct.w=rgb.w; rct.h=rgb.h; rct.c=3;
    rct.px.resize(static_cast<size_t>(rct.w)*rct.h*3);
    for (int i=0;i<rgb.w*rgb.h;++i) {
        int R = rgb.px[3*i+0];
        int G = rgb.px[3*i+1];
        int B = rgb.px[3*i+2];
        int16_t Y = (int16_t)G;
        int16_t U = (int16_t)(R - G); // [-255..255]
        int16_t V = (int16_t)(B - G);
        rct.px[3*i+0]=Y;
        rct.px[3*i+1]=U;
        rct.px[3*i+2]=V;
    }
    return rct;
}

Image rct_to_rgb(const Image16& rct) {
    if (rct.c != 3) throw std::runtime_error("rct_to_rgb expects 3 channels");
    Image rgb; rgb.w=rct.w; rgb.h=rct.h; rgb.c=3;
    rgb.px.resize(static_cast<size_t>(rgb.w)*rgb.h*3);
    for (int i=0;i<rct.w*rct.h;++i) {
        int Y = rct.px[3*i+0];
        int U = rct.px[3*i+1];
        int V = rct.px[3*i+2];
        int R = Y + U;
        int G = Y;
        int B = Y + V;
        rgb.px[3*i+0]=clamp8i(R);
        rgb.px[3*i+1]=clamp8i(G);
        rgb.px[3*i+2]=clamp8i(B);
    }
    return rgb;
}

bool images_equal(const Image& a, const Image& b) {
    return a.w==b.w && a.h==b.h && a.c==b.c && a.px==b.px;
}
