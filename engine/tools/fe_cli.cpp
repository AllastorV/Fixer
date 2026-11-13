#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "fe/engine.h"
#include "../../plugin/match.h"

namespace {

bool read_file(const char* path, std::vector<char>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    out.resize(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    return static_cast<bool>(f.read(out.data(), out.size()));
}

int fail(const char* what, cudaError_t e) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    return 2;
}

}

int match_bake(int argc, char** argv) {
    std::vector<char> text;
    fxm::State st;
    if (argc != 6 && argc != 7) return std::fprintf(stderr, "usage: fe_cli --match-bake state.json strength protect out.bin [size]\n"), 1;
    if (!read_file(argv[2], text) || !fxm::parse_state(std::string(text.begin(), text.end()), st))
        return std::fprintf(stderr, "bad state\n"), 1;
    int n = argc == 7 ? std::atoi(argv[6]) : fxm::kLutSize;
    if (n < 2 || n > 129) return std::fprintf(stderr, "bad size\n"), 1;
    std::vector<float> lut = fxm::bake_lut(st, std::strtod(argv[3], nullptr), std::strtod(argv[4], nullptr), n);
    std::ofstream(argv[5], std::ios::binary).write(reinterpret_cast<const char*>(lut.data()), lut.size() * sizeof(float));
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && !std::strcmp(argv[1], "--match-bake")) return match_bake(argc, argv);
    if (argc < 10) {
        std::fprintf(stderr,
                     "usage: fe_cli params.bin in.raw out.raw width height lut_size under.bin normal.bin over.bin "
                     "[--window x1 y1 x2 y2] [--print print.bin lo hi] [--grain grain.bin] [--match match.bin size] [--iters N]\n");
        return 1;
    }
    bool has_window = false;
    int window[4] = {0, 0, 0, 0}, iters = 0;
    const char* print_path = nullptr;
    float print_lo = 0.0f, print_hi = 0.0f;
    const char* grain_path = nullptr;
    bool grain_mono = false;
    const char* match_path = nullptr;
    int match_n = 0;
    for (int i = 10; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--window") && i + 4 < argc) {
            for (int k = 0; k < 4; ++k) window[k] = std::atoi(argv[++i]);
            has_window = true;
        } else if (!std::strcmp(argv[i], "--print") && i + 3 < argc) {
            print_path = argv[++i];
            print_lo = std::strtof(argv[++i], nullptr);
            print_hi = std::strtof(argv[++i], nullptr);
        } else if (!std::strcmp(argv[i], "--grain") && i + 1 < argc) {
            grain_path = argv[++i];
        } else if (!std::strcmp(argv[i], "--match") && i + 2 < argc) {
            match_path = argv[++i];
            match_n = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--grain-mono")) {
            grain_mono = true;
        } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else {
            return std::fprintf(stderr, "bad option %s\n", argv[i]), 1;
        }
    }
    std::vector<char> pbuf, img;
    if (!read_file(argv[1], pbuf) || pbuf.size() != sizeof(fe::RenderParams)) return std::fprintf(stderr, "bad params\n"), 1;
    fe::RenderParams params;
    std::memcpy(&params, pbuf.data(), sizeof(params));
    int width = std::atoi(argv[4]), height = std::atoi(argv[5]), n = std::atoi(argv[6]);
    size_t bytes = static_cast<size_t>(width) * height * 4 * sizeof(float);
    if (width <= 0 || height <= 0 || !read_file(argv[2], img) || img.size() != bytes) return std::fprintf(stderr, "bad image\n"), 1;
    fe::LutSet luts;
    for (int s = 0; s < 3; ++s) {
        std::vector<char> lut;
        if (!read_file(argv[7 + s], lut) || lut.size() != static_cast<size_t>(n) * n * n * 3 * sizeof(float))
            return std::fprintf(stderr, "bad lut %d\n", s), 1;
        cudaError_t e = luts.upload(s, reinterpret_cast<const float*>(lut.data()), n);
        if (e != cudaSuccess) return fail("upload", e);
    }
    if (print_path) {
        std::vector<char> lut;
        if (!read_file(print_path, lut) || lut.size() != static_cast<size_t>(n) * n * n * 3 * sizeof(float))
            return std::fprintf(stderr, "bad print lut\n"), 1;
        cudaError_t e = luts.upload(fe::LutSet::kPrintSlot, reinterpret_cast<const float*>(lut.data()), n);
        if (e != cudaSuccess) return fail("upload", e);
        luts.set_print_domain(print_lo, print_hi);
    }
    if (grain_path) {
        std::vector<char> g;
        if (!read_file(grain_path, g) || g.size() != 3 * fe::kGrainTableSize * sizeof(float))
            return std::fprintf(stderr, "bad grain table\n"), 1;
        luts.set_grain_table(reinterpret_cast<const float*>(g.data()));
    }
    luts.set_grain_mono(grain_mono);
    if (match_path) {
        std::vector<char> lut;
        if (match_n < 2 || !read_file(match_path, lut) || lut.size() != static_cast<size_t>(match_n) * match_n * match_n * 3 * sizeof(float))
            return std::fprintf(stderr, "bad match lut\n"), 1;
        cudaError_t e = luts.upload(fe::LutSet::kMatchSlot, reinterpret_cast<const float*>(lut.data()), match_n);
        if (e != cudaSuccess) return fail("upload", e);
    }
    float *src = nullptr, *dst = nullptr;
    size_t sp = 0, dp = 0, row = width * 4 * sizeof(float);
    cudaError_t e;
    if ((e = cudaMallocPitch(&src, &sp, row, height)) != cudaSuccess) return fail("malloc", e);
    if ((e = cudaMallocPitch(&dst, &dp, row, height)) != cudaSuccess) return fail("malloc", e);
    if ((e = cudaMemcpy2D(src, sp, img.data(), row, row, height, cudaMemcpyHostToDevice)) != cudaSuccess) return fail("h2d", e);
    std::vector<float> fill(static_cast<size_t>(width) * height * 4, -7.0f);
    if ((e = cudaMemcpy2D(dst, dp, fill.data(), row, row, height, cudaMemcpyHostToDevice)) != cudaSuccess) return fail("h2d", e);
    fe::Rect bounds{params.origin_x, params.origin_y, params.origin_x + width, params.origin_y + height};
    fe::Rect win = bounds;
    if (has_window) win = {window[0], window[1], window[2], window[3]};
    auto run = [&]() {
        return fe::render_region(fe::Image{reinterpret_cast<const char*>(src), sp, bounds}, dst, dp, bounds, win, params, luts,
                                 nullptr);
    };
    if ((e = run()) != cudaSuccess) return fail("render", e);
    if (iters > 0) {
        std::vector<char> lut;
        read_file(argv[7], lut);
        size_t free0 = 0, free1 = 0, total = 0;
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free0, &total);
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        float upload_ms = 0.0f, render_ms = 0.0f;
        for (int i = 0; i < iters; ++i) {
            cudaEventRecord(a);
            if ((e = luts.upload(0, reinterpret_cast<const float*>(lut.data()), n)) != cudaSuccess) return fail("reupload", e);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float t = 0.0f;
            cudaEventElapsedTime(&t, a, b);
            upload_ms += t;
            cudaEventRecord(a);
            if ((e = run()) != cudaSuccess) return fail("render", e);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            cudaEventElapsedTime(&t, a, b);
            render_ms += t;
        }
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free1, &total);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        std::printf("iters=%d render_ms_avg=%.4f lut_upload_ms_avg=%.4f free_before=%zu free_after=%zu delta=%lld\n", iters,
                    render_ms / iters, upload_ms / iters, free0, free1,
                    static_cast<long long>(free0) - static_cast<long long>(free1));
    }
    if ((e = cudaMemcpy2D(img.data(), row, dst, dp, row, height, cudaMemcpyDeviceToHost)) != cudaSuccess) return fail("d2h", e);
    cudaFree(src);
    cudaFree(dst);
    std::ofstream(argv[3], std::ios::binary).write(img.data(), img.size());
    return 0;
}
