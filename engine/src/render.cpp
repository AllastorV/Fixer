#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "fe/engine.h"
#include "kernels_cl.h"
#include "spaces_gen.h"

namespace fe {

namespace {

struct Args {
    RenderParams p;
    float weight[3];
    float to_working[9];
    float from_working[9];
    int32_t two_stage;
    float print_lo;
    float print_inv;
    float grain[3 * kGrainTableSize];
    int32_t grain_mono;
    int32_t match_on;
    int32_t match_size;
};

static_assert(sizeof(Args) == sizeof(RenderParams) + 4 * (3 + 9 + 9 + 3 + 3 * kGrainTableSize + 3), "Args must have no padding");

std::mutex gProgramLock;
std::map<std::pair<cl_context, cl_device_id>, cl_program> gPrograms;
std::string gBuildLog;

std::string defines() {
    std::string s;
    char line[160];
    auto f = [&](const char* name, float v) {
        char num[48];
        *std::to_chars(num, num + sizeof num - 1, v, std::chars_format::scientific, 9).ptr = '\0';
        std::snprintf(line, sizeof line, "#define %s %sf\n", name, num);
        s += line;
    };
    auto i = [&](const char* name, int v) {
        std::snprintf(line, sizeof line, "#define %s %d\n", name, v);
        s += line;
    };
    i("kSpace_dwg_di", kSpace_dwg_di);
    i("kSpace_dwg_linear", kSpace_dwg_linear);
    i("kSpace_acescct", kSpace_acescct);
    i("kSpace_rec709_g24", kSpace_rec709_g24);
    i("kGrainTableSize", kGrainTableSize);
    i("FE_FILM_BYPASS", LutSet::kFilmBypass);
    f("kGrainTableLo", kGrainTableLo);
    f("kGrainTableHi", kGrainTableHi);
    f("kGrainBaseSigmaLog10", kGrainBaseSigmaLog10);
    f("kGrainCloudRadius", kGrainCloudRadius);
    f("kGrainCloudMean", kGrainCloudMean);
    f("kGrainCloudStd", kGrainCloudStd);
    f("kGrainMinBlurPx", kGrainMinBlurPx);
    f("kGrainMaxBlurPx", kGrainMaxBlurPx);
    f("kHalationMix", kHalationMix);
    return s;
}

cl_int check_layout(cl_program prog, cl_context ctx, cl_command_queue q) {
    cl_int e = CL_SUCCESS;
    cl_kernel k = clCreateKernel(prog, "args_size", &e);
    if (e != CL_SUCCESS) return e;
    cl_mem out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(cl_int), nullptr, &e);
    cl_int size = 0;
    if (e == CL_SUCCESS) e = clSetKernelArg(k, 0, sizeof(cl_mem), &out);
    size_t one = 1;
    if (e == CL_SUCCESS) e = clEnqueueNDRangeKernel(q, k, 1, nullptr, &one, nullptr, 0, nullptr, nullptr);
    if (e == CL_SUCCESS) e = clEnqueueReadBuffer(q, out, CL_TRUE, 0, sizeof(cl_int), &size, 0, nullptr, nullptr);
    if (out) clReleaseMemObject(out);
    clReleaseKernel(k);
    if (e == CL_SUCCESS && size != static_cast<cl_int>(sizeof(Args))) {
        gBuildLog = "kernel argument layout differs from the host (" + std::to_string(size) + " vs " + std::to_string(sizeof(Args)) + " bytes)";
        return CL_BUILD_PROGRAM_FAILURE;
    }
    return e;
}

cl_int program_for(cl_command_queue q, cl_program* out, cl_device_id* device) {
    cl_context ctx = nullptr;
    cl_int e = clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof ctx, &ctx, nullptr);
    if (e == CL_SUCCESS) e = clGetCommandQueueInfo(q, CL_QUEUE_DEVICE, sizeof *device, device, nullptr);
    if (e != CL_SUCCESS) return e;
    std::lock_guard<std::mutex> guard(gProgramLock);
    auto key = std::make_pair(ctx, *device);
    auto it = gPrograms.find(key);
    if (it != gPrograms.end()) {
        *out = it->second;
        return CL_SUCCESS;
    }
    std::string source = defines() + kKernelSource;
    const char* text = source.c_str();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &text, nullptr, &e);
    if (e != CL_SUCCESS) return e;
    cl_device_fp_config fp = 0;
    clGetDeviceInfo(*device, CL_DEVICE_SINGLE_FP_CONFIG, sizeof fp, &fp, nullptr);
    std::string options = "-cl-std=CL1.2";
    if (fp & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) options += " -cl-fp32-correctly-rounded-divide-sqrt";
    e = clBuildProgram(prog, 1, device, options.c_str(), nullptr, nullptr);
    if (e != CL_SUCCESS) {
        size_t n = 0;
        clGetProgramBuildInfo(prog, *device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        if (n) clGetProgramBuildInfo(prog, *device, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
        gBuildLog = log.c_str();
        clReleaseProgram(prog);
        return e;
    }
    if ((e = check_layout(prog, ctx, q)) != CL_SUCCESS) {
        clReleaseProgram(prog);
        return e;
    }
    clRetainContext(ctx);
    gPrograms[key] = prog;
    *out = prog;
    return CL_SUCCESS;
}

struct Kernel {
    cl_kernel k = nullptr;
    int next = 0;
    cl_int err = CL_SUCCESS;
    Kernel(cl_program prog, const char* name) { k = clCreateKernel(prog, name, &err); }
    ~Kernel() {
        if (k) clReleaseKernel(k);
    }
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    template <class T>
    Kernel& arg(const T& v) {
        if (err == CL_SUCCESS) err = clSetKernelArg(k, next, sizeof(T), &v);
        ++next;
        return *this;
    }
};

struct Ctx {
    cl_command_queue q;
    cl_context context;
    cl_device_id device;
    cl_program prog;
};

cl_int launch(const Ctx& c, Kernel& kernel, size_t w, size_t h) {
    if (kernel.err != CL_SUCCESS) return kernel.err;
    size_t wg = 0;
    clGetKernelWorkGroupInfo(kernel.k, c.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof wg, &wg, nullptr);
    if (h == 0) {
        size_t local = std::min<size_t>(256, std::max<size_t>(wg, 1));
        size_t global = (w + local - 1) / local * local;
        return clEnqueueNDRangeKernel(c.q, kernel.k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    }
    size_t side = wg >= 256 ? 16 : wg >= 64 ? 8 : 0;
    if (!side) {
        size_t global[2] = {w, h};
        return clEnqueueNDRangeKernel(c.q, kernel.k, 2, nullptr, global, nullptr, 0, nullptr, nullptr);
    }
    size_t local[2] = {side, side}, global[2] = {(w + side - 1) / side * side, (h + side - 1) / side * side};
    return clEnqueueNDRangeKernel(c.q, kernel.k, 2, nullptr, global, local, 0, nullptr, nullptr);
}

void slot_weights(float p, float* w) {
    float t = std::fmin(std::fabs(p), 2.0f) / 2.0f;
    float s = t * t * (3.0f - 2.0f * t);
    w[0] = p < 0.0f ? s : 0.0f;
    w[1] = 1.0f - s;
    w[2] = p >= 0.0f ? s : 0.0f;
}

struct Pool {
    cl_context context;
    std::vector<cl_mem> mems;
    explicit Pool(cl_context c) : context(c) {}
    ~Pool() {
        for (cl_mem m : mems) clReleaseMemObject(m);
    }
    cl_int alloc(cl_mem* out, size_t bytes, const void* init = nullptr) {
        cl_int e = CL_SUCCESS;
        cl_mem_flags flags = init ? CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR : CL_MEM_READ_WRITE;
        *out = clCreateBuffer(context, flags, bytes, const_cast<void*>(init), &e);
        if (e == CL_SUCCESS) mems.push_back(*out);
        return e;
    }
};

std::vector<float> gauss_weights(float sigma, int& r) {
    r = static_cast<int>(std::ceil(3.0f * sigma));
    std::vector<float> k(2 * r + 1);
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) sum += k[i + r] = static_cast<float>(std::exp(-0.5 * (i / (double)sigma) * (i / (double)sigma)));
    for (float& v : k) v = static_cast<float>(v / sum);
    return k;
}

Args make_args(const RenderParams& p, const LutSet& luts) {
    Args a{};
    a.p = p;
    slot_weights(p.push_pull, a.weight);
    a.two_stage = luts.two_stage() ? 1 : 0;
    a.print_lo = luts.print_lo();
    a.print_inv = a.two_stage ? 1.0f / (luts.print_hi() - luts.print_lo()) : 0.0f;
    for (int i = 0; i < 9; ++i) {
        a.to_working[i] = kToWorking[p.in_space][i];
        a.from_working[i] = kFromWorking[p.out_space][i];
    }
    a.grain_mono = luts.grain_mono() ? 1 : 0;
    a.match_on = (p.reserved2[0] & LutSet::kMatchOn) && luts.buffer(LutSet::kMatchSlot) ? 1 : 0;
    a.match_size = p.reserved;
    if (const float* t = luts.grain_table()) {
        for (int i = 0; i < 3 * kGrainTableSize; ++i) a.grain[i] = t[i];
    } else {
        for (int i = 0; i < kGrainTableSize; ++i) {
            double u = (kGrainTableLo + (kGrainTableHi - kGrainTableLo) * i / (kGrainTableSize - 1.0)) / kGrainTonalWidth;
            float v = static_cast<float>(kGrainTonalFloor + (1.0 - kGrainTonalFloor) * std::exp(-0.5 * u * u));
            for (int c = 0; c < 3; ++c) a.grain[c * kGrainTableSize + i] = v;
        }
    }
    return a;
}

bool match_bad(const RenderParams& p, const LutSet& luts) {
    return (p.reserved2[0] & LutSet::kMatchOn) &&
           (!luts.buffer(LutSet::kMatchSlot) || p.reserved < 2 || p.reserved != luts.match_size());
}

int blur_margin(float sigma) { return static_cast<int>(std::ceil(3.0f * std::fmin(std::fmax(sigma, 0.5f), kSpatialMaxSigmaPx))); }

cl_int blur(const Ctx& c, Pool& pool, cl_mem in, cl_mem out, cl_mem tmp, int w, int h, float sigma) {
    sigma = std::fmin(std::fmax(sigma, 0.5f), kSpatialMaxSigmaPx);
    int r = 0;
    std::vector<float> k = gauss_weights(sigma, r);
    cl_mem weights = nullptr;
    cl_int e = pool.alloc(&weights, k.size() * sizeof(float), k.data());
    if (e != CL_SUCCESS) return e;
    Kernel bh(c.prog, "blur_h_clamp"), bv(c.prog, "blur_v_clamp");
    bh.arg(in).arg(tmp).arg(w).arg(h).arg(weights).arg(r);
    bv.arg(tmp).arg(out).arg(w).arg(h).arg(weights).arg(r);
    if ((e = launch(c, bh, w, h)) != CL_SUCCESS) return e;
    return launch(c, bv, w, h);
}

cl_int prepare(cl_command_queue q, const LutSet& luts, Ctx& c) {
    c.q = q;
    cl_int e = program_for(q, &c.prog, &c.device);
    if (e != CL_SUCCESS) return e;
    if ((e = clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof c.context, &c.context, nullptr)) != CL_SUCCESS) return e;
    return luts.context() == c.context ? CL_SUCCESS : CL_INVALID_CONTEXT;
}

bool bad_space(const RenderParams& p) { return p.in_space < 0 || p.in_space >= kSpaceCount || p.out_space < 0 || p.out_space >= kSpaceCount; }

}

LutSet::~LutSet() {
    for (cl_mem m : buf_)
        if (m) clReleaseMemObject(m);
    if (context_) clReleaseContext(context_);
}

void LutSet::borrow_match(const LutSet& from) {
    if (!match_borrowed_ && buf_[kMatchSlot]) return;
    if (from.context_ != context_ || from.buf_[kMatchSlot] == buf_[kMatchSlot]) return;
    if (buf_[kMatchSlot]) clReleaseMemObject(buf_[kMatchSlot]);
    buf_[kMatchSlot] = from.buf_[kMatchSlot];
    if (buf_[kMatchSlot]) clRetainMemObject(buf_[kMatchSlot]);
    match_size_ = from.match_size_;
    match_borrowed_ = true;
}

bool LutSet::ready() const { return size_ > 1 && buf_[0] && buf_[1] && buf_[2]; }

cl_int LutSet::upload(cl_context context, int slot, const float* rgb, int size) {
    if (!context || slot < 0 || slot > kMatchSlot || size < 2 || (slot != kMatchSlot && size_ && size_ != size)) return CL_INVALID_VALUE;
    if (context_ && context_ != context) return CL_INVALID_CONTEXT;
    size_t count = static_cast<size_t>(size) * size * size;
    std::vector<cl_float4> rgba(count);
    for (size_t i = 0; i < count; ++i) rgba[i] = {{rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2], 0.0f}};
    if (buf_[slot]) clReleaseMemObject(buf_[slot]), buf_[slot] = nullptr;
    if (slot == kMatchSlot) match_size_ = 0, match_borrowed_ = false;
    cl_int e = CL_SUCCESS;
    cl_mem m = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, count * sizeof(cl_float4), rgba.data(), &e);
    if (e != CL_SUCCESS) return e;
    if (!context_) clRetainContext(context_ = context);
    buf_[slot] = m;
    (slot == kMatchSlot ? match_size_ : size_) = size;
    return CL_SUCCESS;
}

cl_int render(cl_mem src, size_t src_offset, size_t src_pitch, cl_mem dst, size_t dst_offset, size_t dst_pitch, int width,
              int height, const RenderParams& params, const LutSet& luts, cl_command_queue queue) {
    if (width <= 0 || height <= 0 || bad_space(params) || (src_offset | src_pitch | dst_offset | dst_pitch) % 4 ||
        src_pitch < static_cast<size_t>(width) * 16 || dst_pitch < static_cast<size_t>(width) * 16)
        return CL_INVALID_VALUE;
    if (params.impact == 0.0f) {
        size_t so[3] = {src_offset, 0, 0}, d[3] = {dst_offset, 0, 0}, region[3] = {static_cast<size_t>(width) * 16, static_cast<size_t>(height), 1};
        return clEnqueueCopyBufferRect(queue, src, dst, so, d, region, src_pitch, 0, dst_pitch, 0, 0, nullptr, nullptr);
    }
    if (!luts.ready()) return CL_INVALID_OPERATION;
    if (match_bad(params, luts)) return CL_INVALID_VALUE;
    Ctx c{};
    cl_int e = prepare(queue, luts, c);
    if (e != CL_SUCCESS) return e;
    RenderParams p = params;
    p.lut_size = luts.size();
    Args a = make_args(p, luts);
    Pool pool(c.context);
    cl_mem args = nullptr;
    if ((e = pool.alloc(&args, sizeof a, &a)) != CL_SUCCESS) return e;
    Kernel k(c.prog, "film_kernel");
    k.arg(src).arg(cl_ulong(src_offset / 4)).arg(cl_ulong(src_pitch / 4)).arg(dst).arg(cl_ulong(dst_offset / 4)).arg(cl_ulong(dst_pitch / 4));
    k.arg(width).arg(height).arg(args);
    for (int s = 0; s < 5; ++s) k.arg(luts.buffer(s));
    return launch(c, k, width, height);
}

cl_int render_region(Image src, cl_mem dst, size_t dst_pitch, Rect dst_bounds, Rect win, const RenderParams& params,
                     const LutSet& luts, cl_command_queue queue) {
    win.x1 = std::max({win.x1, src.bounds.x1, dst_bounds.x1});
    win.y1 = std::max({win.y1, src.bounds.y1, dst_bounds.y1});
    win.x2 = std::min({win.x2, src.bounds.x2, dst_bounds.x2});
    win.y2 = std::min({win.y2, src.bounds.y2, dst_bounds.y2});
    if (win.width() <= 0 || win.height() <= 0) return CL_SUCCESS;
    bool spatial = params.impact != 0.0f && (params.halation_amount > 0.0f || params.bloom_amount > 0.0f);
    if (!spatial) {
        RenderParams p = params;
        p.origin_x = win.x1;
        p.origin_y = win.y1;
        size_t so = src.offset + static_cast<size_t>(win.y1 - src.bounds.y1) * src.pitch + static_cast<size_t>(win.x1 - src.bounds.x1) * 16;
        size_t d = static_cast<size_t>(win.y1 - dst_bounds.y1) * dst_pitch + static_cast<size_t>(win.x1 - dst_bounds.x1) * 16;
        return render(src.mem, so, src.pitch, dst, d, dst_pitch, win.width(), win.height(), p, luts, queue);
    }
    if (bad_space(params) || (src.offset | src.pitch | dst_pitch) % 4) return CL_INVALID_VALUE;
    if (!luts.ready()) return CL_INVALID_OPERATION;
    if (match_bad(params, luts)) return CL_INVALID_VALUE;
    Ctx c{};
    cl_int err = prepare(queue, luts, c);
    if (err != CL_SUCCESS) return err;
    RenderParams p = params;
    p.lut_size = luts.size();
    Args a = make_args(p, luts);
    auto reach = [&](float sigma) { return p.spatial_half ? 2 * blur_margin(0.5f * sigma) + 3 : blur_margin(sigma); };
    int m = (p.halation_amount > 0.0f ? reach(p.halation_radius_px) : 0) + (p.bloom_amount > 0.0f ? reach(p.bloom_radius_px) : 0);
    Rect e{std::max(win.x1 - m, src.bounds.x1), std::max(win.y1 - m, src.bounds.y1), std::min(win.x2 + m, src.bounds.x2),
           std::min(win.y2 + m, src.bounds.y2)};
    bool half = p.spatial_half != 0;
    if (half) {
        e.x1 -= (e.x1 - src.bounds.x1) & 1;
        e.y1 -= (e.y1 - src.bounds.y1) & 1;
        if (((e.x2 - e.x1) & 1) && e.x2 < src.bounds.x2) ++e.x2;
        if (((e.y2 - e.y1) & 1) && e.y2 < src.bounds.y2) ++e.y2;
    }
    int w = e.width(), h = e.height();
    size_t n = static_cast<size_t>(w) * h;
    int w2 = (w + 1) / 2, h2 = (h + 1) / 2;
    size_t nb = half ? static_cast<size_t>(w2) * h2 : n;
    Pool pool(c.context);
    cl_mem args = nullptr, lin = nullptr, t1 = nullptr, t2 = nullptr, tmp = nullptr, lh = nullptr;
    const size_t px = sizeof(cl_float4);
    if ((err = pool.alloc(&args, sizeof a, &a)) != CL_SUCCESS || (err = pool.alloc(&lin, n * px)) != CL_SUCCESS ||
        (err = pool.alloc(&t1, nb * px)) != CL_SUCCESS || (err = pool.alloc(&t2, nb * px)) != CL_SUCCESS ||
        (err = pool.alloc(&tmp, nb * px)) != CL_SUCCESS || (half && (err = pool.alloc(&lh, nb * px)) != CL_SUCCESS))
        return err;
    cl_ulong so = src.offset / 4, sp = src.pitch / 4;
    Kernel prep(c.prog, "region_prepare");
    prep.arg(src.mem).arg(so).arg(sp).arg(src.bounds.x1).arg(src.bounds.y1).arg(e.x1).arg(e.y1).arg(w).arg(h).arg(lin).arg(args);
    if ((err = launch(c, prep, w, h)) != CL_SUCCESS) return err;
    int count = static_cast<int>(n);
    if (p.halation_amount > 0.0f) {
        if (half) {
            Kernel down(c.prog, "downsample2"), comb(c.prog, "halation_combine_half");
            down.arg(lin).arg(w).arg(h).arg(lh).arg(w2).arg(h2);
            if ((err = launch(c, down, w2, h2)) != CL_SUCCESS) return err;
            if ((err = blur(c, pool, lh, t1, tmp, w2, h2, 0.5f * kHalationNarrow * p.halation_radius_px)) != CL_SUCCESS) return err;
            if ((err = blur(c, pool, lh, t2, tmp, w2, h2, 0.5f * p.halation_radius_px)) != CL_SUCCESS) return err;
            comb.arg(lin).arg(t1).arg(t2).arg(w).arg(h).arg(w2).arg(h2).arg(p.halation_amount);
            if ((err = launch(c, comb, w, h)) != CL_SUCCESS) return err;
        } else {
            if ((err = blur(c, pool, lin, t1, tmp, w, h, kHalationNarrow * p.halation_radius_px)) != CL_SUCCESS) return err;
            if ((err = blur(c, pool, lin, t2, tmp, w, h, p.halation_radius_px)) != CL_SUCCESS) return err;
            Kernel comb(c.prog, "halation_combine");
            comb.arg(lin).arg(t1).arg(t2).arg(count).arg(p.halation_amount);
            if ((err = launch(c, comb, n, 0)) != CL_SUCCESS) return err;
        }
    }
    Kernel film(c.prog, "region_film");
    film.arg(lin).arg(e.x1).arg(e.y1).arg(w).arg(h).arg(args);
    for (int s = 0; s < 5; ++s) film.arg(luts.buffer(s));
    if ((err = launch(c, film, w, h)) != CL_SUCCESS) return err;
    if (p.bloom_amount > 0.0f) {
        if (half) {
            Kernel down(c.prog, "downsample2"), comb(c.prog, "bloom_combine_half");
            down.arg(lin).arg(w).arg(h).arg(lh).arg(w2).arg(h2);
            if ((err = launch(c, down, w2, h2)) != CL_SUCCESS) return err;
            if ((err = blur(c, pool, lh, t1, tmp, w2, h2, 0.5f * p.bloom_radius_px)) != CL_SUCCESS) return err;
            comb.arg(lin).arg(t1).arg(w).arg(h).arg(w2).arg(h2).arg(p.bloom_amount);
            if ((err = launch(c, comb, w, h)) != CL_SUCCESS) return err;
        } else {
            if ((err = blur(c, pool, lin, t1, tmp, w, h, p.bloom_radius_px)) != CL_SUCCESS) return err;
            Kernel comb(c.prog, "bloom_combine");
            comb.arg(lin).arg(t1).arg(count).arg(p.bloom_amount);
            if ((err = launch(c, comb, n, 0)) != CL_SUCCESS) return err;
        }
    }
    Kernel fin(c.prog, "region_final");
    fin.arg(src.mem).arg(so).arg(sp).arg(src.bounds.x1).arg(src.bounds.y1).arg(e.x1).arg(e.y1).arg(w).arg(win.x1).arg(win.y1);
    fin.arg(win.width()).arg(win.height()).arg(lin).arg(dst).arg(cl_ulong(dst_pitch / 4)).arg(dst_bounds.x1).arg(dst_bounds.y1).arg(args);
    return launch(c, fin, win.width(), win.height());
}

const char* error_string(cl_int e) {
    switch (e) {
        case CL_SUCCESS: return "success";
        case CL_OUT_OF_RESOURCES: return "OpenCL: out of resources";
        case CL_OUT_OF_HOST_MEMORY: return "OpenCL: out of host memory";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "OpenCL: GPU memory allocation failed";
        case CL_INVALID_BUFFER_SIZE: return "OpenCL: buffer larger than the GPU allows";
        case CL_INVALID_VALUE: return "OpenCL: invalid value";
        case CL_INVALID_CONTEXT: return "OpenCL: resources belong to another GPU context";
        case CL_INVALID_COMMAND_QUEUE: return "OpenCL: invalid command queue";
        case CL_INVALID_MEM_OBJECT: return "OpenCL: invalid image buffer";
        case CL_INVALID_OPERATION: return "OpenCL: film stock not loaded";
        case CL_BUILD_PROGRAM_FAILURE: return "OpenCL: kernel build failed";
        case CL_INVALID_WORK_GROUP_SIZE: return "OpenCL: invalid work-group size";
        default: {
            thread_local char text[48];
            std::snprintf(text, sizeof text, "OpenCL error %d", static_cast<int>(e));
            return text;
        }
    }
}

std::string build_log() {
    std::lock_guard<std::mutex> guard(gProgramLock);
    return gBuildLog;
}

bool out_of_memory(cl_int e) {
    return e == CL_MEM_OBJECT_ALLOCATION_FAILURE || e == CL_OUT_OF_RESOURCES || e == CL_OUT_OF_HOST_MEMORY || e == CL_INVALID_BUFFER_SIZE;
}

}
