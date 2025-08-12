#include "residualIO.h"
#include <fstream>
#include <stdexcept>

static constexpr uint32_t MAGIC = 0x52313652; // Residual int16 Raw

// -------- Save here --------
void save_residuals(const std::string& path,
                    int mode, int w, int h, int c,
                    const std::vector<int16_t>& residuals)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for write: " + path);
    uint32_t magic = MAGIC;
    int32_t  imode = mode, iw=w, ih=h, ic=c;
    int64_t  count = static_cast<int64_t>(residuals.size());

    f.write(reinterpret_cast<char*>(&magic), 4);
    f.write(reinterpret_cast<char*>(&imode), 4);
    f.write(reinterpret_cast<char*>(&iw), 4);
    f.write(reinterpret_cast<char*>(&ih), 4);
    f.write(reinterpret_cast<char*>(&ic), 4);
    f.write(reinterpret_cast<char*>(&count), 8);
    f.write(reinterpret_cast<const char*>(residuals.data()), count * sizeof(int16_t));
    if (!f) throw std::runtime_error("Write failed: " + path);
}
// -------- Load here --------
ResidualFile load_residuals(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open for read: " + path);

    uint32_t magic; f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != MAGIC) throw std::runtime_error("Bad residual file");

    ResidualFile rf{};
    int64_t count=0;
    f.read(reinterpret_cast<char*>(&rf.mode), 4);
    f.read(reinterpret_cast<char*>(&rf.w), 4);
    f.read(reinterpret_cast<char*>(&rf.h), 4);
    f.read(reinterpret_cast<char*>(&rf.c), 4);
    f.read(reinterpret_cast<char*>(&count), 8);

    if (rf.w<=0 || rf.h<=0 || rf.c<=0) throw std::runtime_error("Invalid residual metadata");
    rf.residuals.resize(static_cast<size_t>(count));

    f.read(reinterpret_cast<char*>(rf.residuals.data()), count * sizeof(int16_t));
    if (!f) throw std::runtime_error("Read failed: " + path);
    return rf;
}
