#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "cl_gpu.h"
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

int fail(const char* what, cl_int e) {
    std::fprintf(stderr, "%s: %s\n", what, fe::error_string(e));
    if (e == CL_BUILD_PROGRAM_FAILURE) std::fprintf(stderr, "%s\n", fe::build_log().c_str());
    return 2;
}

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
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
    ClGpu gpu;
    fe::LutSet luts;
    for (int s = 0; s < 3; ++s) {
        std::vector<char> lut;
        if (!read_file(argv[7 + s], lut) || lut.size() != static_cast<size_t>(n) * n * n * 3 * sizeof(float))
            return std::fprintf(stderr, "bad lut %d\n", s), 1;
        cl_int e = luts.upload(gpu.context, s, reinterpret_cast<const float*>(lut.data()), n);
        if (e != CL_SUCCESS) return fail("upload", e);
    }
    if (print_path) {
        std::vector<char> lut;
        if (!read_file(print_path, lut) || lut.size() != static_cast<size_t>(n) * n * n * 3 * sizeof(float))
            return std::fprintf(stderr, "bad print lut\n"), 1;
        cl_int e = luts.upload(gpu.context, fe::LutSet::kPrintSlot, reinterpret_cast<const float*>(lut.data()), n);
        if (e != CL_SUCCESS) return fail("upload", e);
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
        cl_int e = luts.upload(gpu.context, fe::LutSet::kMatchSlot, reinterpret_cast<const float*>(lut.data()), match_n);
        if (e != CL_SUCCESS) return fail("upload", e);
    }
    size_t row = static_cast<size_t>(width) * 4 * sizeof(float);
    std::vector<float> fill(static_cast<size_t>(width) * height * 4, -7.0f);
    cl_mem src = gpu.buffer(bytes, img.data()), dst = gpu.buffer(bytes, fill.data());
    fe::Rect bounds{params.origin_x, params.origin_y, params.origin_x + width, params.origin_y + height};
    fe::Rect win = bounds;
    if (has_window) win = {window[0], window[1], window[2], window[3]};
    auto run = [&]() { return fe::render_region(fe::Image{src, 0, row, bounds}, dst, row, bounds, win, params, luts, gpu.queue); };
    cl_int e;
    if ((e = run()) != CL_SUCCESS) return fail("render", e);
    if ((e = clFinish(gpu.queue)) != CL_SUCCESS) return fail("finish", e);
    if (iters > 0) {
        std::vector<char> lut;
        read_file(argv[7], lut);
        double upload_ms = 0.0, render_ms = 0.0;
        for (int i = 0; i < iters; ++i) {
            auto t = std::chrono::steady_clock::now();
            if ((e = luts.upload(gpu.context, 0, reinterpret_cast<const float*>(lut.data()), n)) != CL_SUCCESS) return fail("reupload", e);
            upload_ms += ms_since(t);
            t = std::chrono::steady_clock::now();
            if ((e = run()) != CL_SUCCESS) return fail("render", e);
            clFinish(gpu.queue);
            render_ms += ms_since(t);
        }
        std::printf("iters=%d render_ms_avg=%.4f lut_upload_ms_avg=%.4f\n", iters, render_ms / iters, upload_ms / iters);
    }
    if ((e = clEnqueueReadBuffer(gpu.queue, dst, CL_TRUE, 0, bytes, img.data(), 0, nullptr, nullptr)) != CL_SUCCESS) return fail("d2h", e);
    clReleaseMemObject(src);
    clReleaseMemObject(dst);
    std::ofstream(argv[3], std::ios::binary).write(img.data(), img.size());
    return 0;
}
