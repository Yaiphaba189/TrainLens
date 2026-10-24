#include "png.hpp"

#define GL_SILENCE_DEPRECATION
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#include <zlib.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

void put_u32(std::vector<unsigned char>& out, uint32_t v) {
    out.push_back((unsigned char)(v >> 24));
    out.push_back((unsigned char)(v >> 16));
    out.push_back((unsigned char)(v >> 8));
    out.push_back((unsigned char)v);
}

void put_chunk(std::vector<unsigned char>& out, const char* type, const std::vector<unsigned char>& data) {
    put_u32(out, (uint32_t)data.size());
    const size_t start = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    put_u32(out, (uint32_t)crc32(0L, out.data() + start, (uInt)(out.size() - start)));
}

}  // namespace

bool save_framebuffer_png(const std::string& path, int w, int h, std::string& err) {
    std::vector<unsigned char> px((size_t)w * h * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());

    // PNG rows run top to bottom, each starting with filter type 0.
    std::vector<unsigned char> raw;
    raw.reserve((size_t)(w * 3 + 1) * h);
    for (int y = h - 1; y >= 0; --y) {
        raw.push_back(0);
        raw.insert(raw.end(), px.begin() + (size_t)y * w * 3, px.begin() + (size_t)(y + 1) * w * 3);
    }
    uLongf zlen = compressBound((uLong)raw.size());
    std::vector<unsigned char> z(zlen);
    if (compress2(z.data(), &zlen, raw.data(), (uLong)raw.size(), 6) != Z_OK) {
        err = "compression failed";
        return false;
    }
    z.resize(zlen);

    std::vector<unsigned char> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    std::vector<unsigned char> ihdr;
    put_u32(ihdr, (uint32_t)w);
    put_u32(ihdr, (uint32_t)h);
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});  // 8-bit RGB, no interlace
    put_chunk(png, "IHDR", ihdr);
    put_chunk(png, "IDAT", z);
    put_chunk(png, "IEND", {});

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        err = "cannot write " + path + ": " + std::strerror(errno);
        return false;
    }
    const bool ok = std::fwrite(png.data(), 1, png.size(), f) == png.size();
    std::fclose(f);
    if (!ok) err = "short write to " + path;
    return ok;
}
