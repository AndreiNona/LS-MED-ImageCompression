#include "imageIO.h"
#include <stdexcept>
#include <algorithm>

// stb implementations ONLY here
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>

#include "stb_image.h"
#include "stb_image_write.h"

//------- Prevent overflow -------
static inline unsigned char clamp8(int v) {
    return (unsigned char)std::clamp(v, 0, 255);
}
static inline unsigned char clamp8i(int v) { //named for int inputs
    return clamp8(v);
}

//------- Load image here -------
Image load_image(const std::string& path) {
    Image im;
    int w,h,c;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 0);
    if (!data) throw std::runtime_error("Failed to load image: " + path);

    im.w=w; im.h=h; im.c=c;
    im.px.assign(data, data + static_cast<size_t>(w)*h*c);
    stbi_image_free(data);

    // Drop alpha to RGB
    if (im.c == 4) {
        Image tmp; tmp.w=im.w; tmp.h=im.h; tmp.c=3;
        tmp.px.resize(static_cast<size_t>(im.w)*im.h*3);
        for (int i=0;i<im.w*im.h;++i) {
            tmp.px[3*i+0]=im.px[4*i+0];
            tmp.px[3*i+1]=im.px[4*i+1];
            tmp.px[3*i+2]=im.px[4*i+2];
        }
        im = std::move(tmp);
    }

    if (im.c!=1 && im.c!=3)
        throw std::runtime_error("Only Gray or RGB images are supported");
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
//------- Save image here -------
void save_png(const std::string& path, const Image& im_in) {
    Image im = im_in;            // make a local copy
    ensure_gray_or_rgb(im);      // enforce 1 or 3 channels
    const int stride = im.w * im.c;
    if (!stbi_write_png(path.c_str(), im.w, im.h, im.c, im.px.data(), stride))
        throw std::runtime_error("Failed to write PNG: " + path);
    std::cout << "Saving " << path << " (" << im.w << "x" << im.h << ", ch=" << im.c << ")\n";
}

// -------- Non-reversible conversions  --------
//Non-reversible because of multiplication by non int and rounding
Image rgb_to_yuv(const Image& rgb) {
    if (rgb.c < 3) throw std::runtime_error("rgb_to_yuv expects RGB");
    Image yuv; yuv.w=rgb.w; yuv.h=rgb.h; yuv.c=3;
    yuv.px.resize(static_cast<size_t>(rgb.w)*rgb.h*3);
    for (int i=0;i<rgb.w*rgb.h;++i) {
        int r = rgb.px[3*i+0], g = rgb.px[3*i+1], b = rgb.px[3*i+2];
        int Y = int(0.299*r + 0.587*g + 0.114*b + 0.5);
        int U = int(-0.169*r - 0.331*g + 0.5*b + 128 + 0.5);
        int V = int( 0.5*r  - 0.419*g - 0.081*b + 128 + 0.5);
        yuv.px[3*i+0]=clamp8(Y);
        yuv.px[3*i+1]=clamp8(U);
        yuv.px[3*i+2]=clamp8(V);
    }
    return yuv;
}

Image yuv_to_rgb(const Image& yuv) {
    if (yuv.c != 3) throw std::runtime_error("yuv_to_rgb expects 3 channels");
    Image rgb; rgb.w=yuv.w; rgb.h=yuv.h; rgb.c=3;
    rgb.px.resize(static_cast<size_t>(yuv.w)*yuv.h*3);
    for (int i=0;i<yuv.w*yuv.h;++i) {
        int Y = yuv.px[3*i+0];
        int U = yuv.px[3*i+1] - 128;
        int V = yuv.px[3*i+2] - 128;
        int r = int(Y + 1.402 * V + 0.5);
        int g = int(Y - 0.344136 * U - 0.714136 * V + 0.5);
        int b = int(Y + 1.772 * U + 0.5);
        rgb.px[3*i+0]=clamp8i(r);
        rgb.px[3*i+1]=clamp8i(g);
        rgb.px[3*i+2]=clamp8i(b);
    }
    return rgb;
}

// -------- Reversible --------
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
