#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

#include "color.cuh"
#include "fe/engine.h"

namespace fe {

namespace {

struct SlotSet {
    cudaTextureObject_t tex[3];
    float weight[3];
    float to_working[9];
    float from_working[9];
    cudaTextureObject_t print_tex;
    int two_stage;
    float print_lo;
    float print_inv;
    float grain[3 * kGrainTableSize];
    int grain_mono;
    cudaTextureObject_t match_tex;
    int match_size;
};

__device__ __forceinline__ float4 texel(cudaTextureObject_t tex, int r, int g, int b) {
    return tex3D<float4>(tex, r + 0.5f, g + 0.5f, b + 0.5f);
}

__device__ __forceinline__ float4 lerp4(float4 a, float4 b, float t) {
    return make_float4(fmaf(t, b.x - a.x, a.x), fmaf(t, b.y - a.y, a.y), fmaf(t, b.z - a.z, a.z), 0.0f);
}

__device__ float3 sample_raw(cudaTextureObject_t tex, float3 c, int n) {
    float pr = fminf(fmaxf(c.x, 0.0f), 1.0f) * (n - 1);
    float pg = fminf(fmaxf(c.y, 0.0f), 1.0f) * (n - 1);
    float pb = fminf(fmaxf(c.z, 0.0f), 1.0f) * (n - 1);
    int r = min(static_cast<int>(pr), n - 2), g = min(static_cast<int>(pg), n - 2), b = min(static_cast<int>(pb), n - 2);
    float fr = pr - r, fg = pg - g, fb = pb - b;
    float4 c0 = lerp4(lerp4(texel(tex, r, g, b), texel(tex, r + 1, g, b), fr),
                      lerp4(texel(tex, r, g + 1, b), texel(tex, r + 1, g + 1, b), fr), fg);
    float4 c1 = lerp4(lerp4(texel(tex, r, g, b + 1), texel(tex, r + 1, g, b + 1), fr),
                      lerp4(texel(tex, r, g + 1, b + 1), texel(tex, r + 1, g + 1, b + 1), fr), fg);
    float4 v = lerp4(c0, c1, fb);
    return make_float3(v.x, v.y, v.z);
}

// Tetrahedral interpolation follows the neutral diagonal without off-axis colour samples.
__device__ float3 sample_match(cudaTextureObject_t tex, float3 lin, int n) {
    float encoded[3] = {di_encode(lin.x), di_encode(lin.y), di_encode(lin.z)};
    int corner[3], order[3] = {0, 1, 2};
    float fraction[3];
    for (int c = 0; c < 3; ++c) {
        float position = fminf(fmaxf(encoded[c], 0.0f), 1.0f) * (n - 1);
        corner[c] = min(static_cast<int>(position), n - 2);
        fraction[c] = position - corner[c];
    }
    for (int i = 0; i < 2; ++i)
        for (int j = i + 1; j < 3; ++j)
            if (fraction[order[j]] > fraction[order[i]]) {
                int swap = order[i]; order[i] = order[j]; order[j] = swap;
            }
    float4 previous = texel(tex, corner[0], corner[1], corner[2]);
    float3 result = make_float3(previous.x, previous.y, previous.z);
    for (int i = 0; i < 3; ++i) {
        ++corner[order[i]];
        float4 next = texel(tex, corner[0], corner[1], corner[2]);
        float weight = fraction[order[i]];
        result.x += weight * (next.x - previous.x);
        result.y += weight * (next.y - previous.y);
        result.z += weight * (next.z - previous.z);
        previous = next;
    }
    return make_float3(di_decode(result.x), di_decode(result.y), di_decode(result.z));
}

__device__ float3 sample_slot(cudaTextureObject_t tex, float3 lin, int n) {
    float3 v = sample_raw(tex, make_float3(di_encode(lin.x), di_encode(lin.y), di_encode(lin.z)), n);
    return make_float3(di_decode(v.x), di_decode(v.y), di_decode(v.z));
}

__device__ __forceinline__ float3 grain_offset(float3 scene, float4 n, float amount, const SlotSet& slots);

__device__ float3 film_eval(float3 lin, const RenderParams& p, const SlotSet& slots, const float4* n) {
    float3 acc = make_float3(0.0f, 0.0f, 0.0f);
    float3 c = make_float3(di_encode(lin.x), di_encode(lin.y), di_encode(lin.z));
    for (int i = 0; i < 3; ++i) {
        if (slots.weight[i] == 0.0f) continue;
        float3 s = slots.two_stage ? sample_raw(slots.tex[i], c, p.lut_size) : sample_slot(slots.tex[i], lin, p.lut_size);
        acc.x += slots.weight[i] * s.x;
        acc.y += slots.weight[i] * s.y;
        acc.z += slots.weight[i] * s.z;
    }
    if (n) {
        float3 g = grain_offset(lin, *n, p.grain_amount, slots);
        if (!slots.two_stage) return make_float3(acc.x * exp10f(-g.x), acc.y * exp10f(-g.y), acc.z * exp10f(-g.z));
        acc = make_float3(acc.x - g.x, acc.y - g.y, acc.z - g.z);
    }
    if (!slots.two_stage) return acc;
    float3 q = make_float3((acc.x + p.print_log_offset[0] - slots.print_lo) * slots.print_inv,
                           (acc.y + p.print_log_offset[1] - slots.print_lo) * slots.print_inv,
                           (acc.z + p.print_log_offset[2] - slots.print_lo) * slots.print_inv);
    float3 v = sample_raw(slots.print_tex, q, p.lut_size);
    return make_float3(di_decode(v.x), di_decode(v.y), di_decode(v.z));
}

__device__ float skin_smooth(float e0, float e1, float x) {
    float t = fminf(fmaxf((x - e0) / (e1 - e0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__device__ float lab_f(float t) { return t > 0.008856451679f ? cbrtf(t) : 7.787037037f * t + 0.137931034f; }

__device__ float skin_weight(float3 c) {
    float r = 1.8986149f * c.x - 0.79217618f * c.y - 0.10643872f * c.z;
    float g = -0.16894879f * c.x + 1.48897575f * c.y - 0.32002697f * c.z;
    float b = -0.12153916f * c.x - 0.31567585f * c.y + 1.43721501f * c.z;
    float fx = lab_f((0.41239080f * r + 0.35758434f * g + 0.18048079f * b) / 0.95045593f);
    float fy = lab_f(0.21263901f * r + 0.71516868f * g + 0.07219232f * b);
    float fz = lab_f((0.01933082f * r + 0.11919478f * g + 0.95053215f * b) / 1.08905775f);
    float L = 116.0f * fy - 16.0f, A = 500.0f * (fx - fy), B = 200.0f * (fy - fz);
    if (!(L >= 15.0f && L <= 95.0f)) return 0.0f;
    float C = hypotf(A, B), h = atan2f(B, A) * 57.29577951f;
    if (h < 0.0f) h += 360.0f;
    return skin_smooth(10.0f, 20.0f, h) * (1.0f - skin_smooth(75.0f, 85.0f, h)) * skin_smooth(2.0f, 8.0f, C) *
           (1.0f - skin_smooth(45.0f, 51.0f, C));
}

__device__ float3 film_stage(float3 lin, const RenderParams& p, const SlotSet& slots, const float4* n) {
    float3 film = (p.reserved2[0] & LutSet::kFilmBypass) ? lin : film_eval(lin, p, slots, n);
    if (!slots.match_tex) return film;
    float3 m = sample_match(slots.match_tex, film, slots.match_size);
    if (!(p.match_protect > 0.0f)) return m;
    float w = 1.0f - fminf(p.match_protect, 1.0f) * skin_weight(film);
    return make_float3(film.x + w * (m.x - film.x), film.y + w * (m.y - film.y), film.z + w * (m.z - film.z));
}

__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ constexpr float kGrainLayerScale[3] = {1.0f, 0.9f, 1.2f};
__device__ constexpr float kGrainCloudCdf[4] = {0.36787944f, 0.73575888f, 0.91969860f, 0.98101184f};

__device__ __forceinline__ float unit24(uint32_t h) { return ((h >> 8) + 0.5f) * 5.9604645e-8f; }

__device__ float clouds(uint32_t base, float px, float py, float radius) {
    float ir = 1.0f / radius, qx = px * ir, qy = py * ir;
    int cx0 = static_cast<int>(floorf(qx)), cy0 = static_cast<int>(floorf(qy));
    float lx = qx - cx0, ly = qy - cy0, acc = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            uint32_t h = hash32(base ^ (static_cast<uint32_t>(cx0 + dx) * 0x8DA6B343u) ^ (static_cast<uint32_t>(cy0 + dy) * 0xD8163841u));
            float u = unit24(h);
            int count = (u > kGrainCloudCdf[0]) + (u > kGrainCloudCdf[1]) + (u > kGrainCloudCdf[2]) + (u > kGrainCloudCdf[3]);
            for (int j = 0; j < count; ++j) {
                h = hash32(h + 0x9E3779B9u);
                float ex = lx - (dx + ((h & 2047u) + 0.5f) * (1.0f / 2048.0f));
                float ey = ly - (dy + (((h >> 11) & 2047u) + 0.5f) * (1.0f / 2048.0f));
                float a = ((h >> 22) + 0.5f) * (2.0f / 1024.0f);
                float b = fmaxf(1.0f - (ex * ex + ey * ey), 0.0f);
                acc = fmaf(a, b * b, acc);
            }
        }
    }
    return (acc - kGrainCloudMean) * (1.0f / kGrainCloudStd);
}

__device__ float4 grain_noise(const RenderParams& p, int gx, int gy, int mono) {
    float sigma = fminf(fmaxf(p.grain_sigma_px, kGrainMinBlurPx), kGrainMaxBlurPx);
    float px = gx + 0.5f, py = gy + 0.5f, r = kGrainCloudRadius * sigma;
    uint32_t b = hash32(hash32(p.grain_seed) ^ static_cast<uint32_t>(p.frame));
    float n0 = clouds(hash32(b ^ 0u), px, py, r * kGrainLayerScale[0]);
    float n1 = mono ? n0 : clouds(hash32(b ^ 1u), px, py, r * kGrainLayerScale[1]);
    float n2 = mono ? n0 : clouds(hash32(b ^ 2u), px, py, r * kGrainLayerScale[2]);
    return make_float4(n0, n1, n2, 0.0f);
}

__device__ __forceinline__ float grain_lookup(const float* t, float v) {
    float u = log10f(fmaxf(v, 9.5367432e-7f) / 0.18f);
    float f = (fminf(fmaxf(u, kGrainTableLo), kGrainTableHi) - kGrainTableLo) *
              ((kGrainTableSize - 1) / (kGrainTableHi - kGrainTableLo));
    int i = min(static_cast<int>(f), kGrainTableSize - 2);
    return fmaf(f - i, t[i + 1] - t[i], t[i]);
}

__device__ __forceinline__ float3 grain_offset(float3 scene, float4 n, float amount, const SlotSet& slots) {
    float k = amount * kGrainBaseSigmaLog10;
    if (slots.grain_mono) {
        float s = k * grain_lookup(slots.grain + kGrainTableSize, scene.y) * n.x;
        return make_float3(s, s, s);
    }
    return make_float3(k * grain_lookup(slots.grain, scene.x) * n.x, k * grain_lookup(slots.grain + kGrainTableSize, scene.y) * n.y,
                       k * grain_lookup(slots.grain + 2 * kGrainTableSize, scene.z) * n.z);
}

__device__ inline float premult_alpha(float a) { return isfinite(a) ? a : (a > 0.0f ? 1.0f : 0.0f); }

__global__ void film_kernel(const char* src, size_t src_pitch, char* dst, size_t dst_pitch, int width, int height,
                            RenderParams p, SlotSet slots) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    float4 px = reinterpret_cast<const float4*>(src + y * src_pitch)[x];
    float3 rgb = make_float3(sanitize(px.x), sanitize(px.y), sanitize(px.z));
    if (p.premultiplied && px.w > 0.0f && isfinite(px.w)) rgb = make_float3(rgb.x / px.w, rgb.y / px.w, rgb.z / px.w);
    float3 dec = make_float3(sanitize(decode(p.in_space, rgb.x)), sanitize(decode(p.in_space, rgb.y)),
                             sanitize(decode(p.in_space, rgb.z)));
    float3 w = mul(slots.to_working, dec);
    w = make_float3(sanitize(w.x), sanitize(w.y), sanitize(w.z));
    float3 r = mul(slots.from_working, w);
    float3 ref = make_float3(sanitize(encode(p.out_space, r.x)), sanitize(encode(p.out_space, r.y)),
                             sanitize(encode(p.out_space, r.z)));
    float g = exp2f(p.exposure_ev);
    float3 lin = make_float3(w.x * g, w.y * g, w.z * g);
    float4 nz;
    if (p.grain_amount > 0.0f) nz = grain_noise(p, p.origin_x + x, p.origin_y + y, slots.grain_mono);
    float3 film = film_stage(lin, p, slots, p.grain_amount > 0.0f ? &nz : nullptr);
    float3 fo = mul(slots.from_working, film);
    fo = make_float3(encode(p.out_space, fo.x), encode(p.out_space, fo.y), encode(p.out_space, fo.z));
    float4 out = make_float4(ref.x + p.impact * (fo.x - ref.x), ref.y + p.impact * (fo.y - ref.y),
                             ref.z + p.impact * (fo.z - ref.z), px.w);
    if (p.premultiplied) out = make_float4(out.x * premult_alpha(px.w), out.y * premult_alpha(px.w), out.z * premult_alpha(px.w), px.w);
    reinterpret_cast<float4*>(dst + y * dst_pitch)[x] = out;
}

void slot_weights(float p, float* w) {
    float t = fminf(fabsf(p), 2.0f) / 2.0f;
    float s = smoothstep01(t);
    w[0] = p < 0.0f ? s : 0.0f;
    w[1] = 1.0f - s;
    w[2] = p >= 0.0f ? s : 0.0f;
}


struct Pool {
    cudaStream_t stream;
    std::vector<void*> ptrs;
    explicit Pool(cudaStream_t s) : stream(s) {}
    ~Pool() {
        for (void* q : ptrs) cudaFreeAsync(q, stream);
    }
    template <class T>
    cudaError_t alloc(T** out, size_t count) {
        void* q = nullptr;
        cudaError_t e = cudaMallocAsync(&q, count * sizeof(T), stream);
        if (e == cudaSuccess) ptrs.push_back(q);
        *out = static_cast<T*>(q);
        return e;
    }
};

std::vector<float> gauss_weights(float sigma, int& r, double* sq) {
    r = static_cast<int>(ceilf(3.0f * sigma));
    std::vector<float> k(2 * r + 1);
    double sum = 0.0;
    for (int i = -r; i <= r; ++i) sum += k[i + r] = static_cast<float>(std::exp(-0.5 * (i / (double)sigma) * (i / (double)sigma)));
    double q = 0.0;
    for (float& v : k) v = static_cast<float>(v / sum), q += static_cast<double>(v) * v;
    if (sq) *sq = q;
    return k;
}

cudaError_t upload_weights(Pool& pool, const std::vector<float>& k, float** out) {
    cudaError_t e = pool.alloc(out, k.size());
    if (e != cudaSuccess) return e;
    return cudaMemcpyAsync(*out, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice, pool.stream);
}

SlotSet make_slots(const RenderParams& p, const LutSet& luts) {
    SlotSet slots;
    slot_weights(p.push_pull, slots.weight);
    for (int i = 0; i < 3; ++i) slots.tex[i] = luts.texture(i);
    slots.two_stage = luts.two_stage() ? 1 : 0;
    slots.print_tex = luts.texture(LutSet::kPrintSlot);
    slots.print_lo = luts.print_lo();
    slots.print_inv = slots.two_stage ? 1.0f / (luts.print_hi() - luts.print_lo()) : 0.0f;
    for (int i = 0; i < 9; ++i) {
        slots.to_working[i] = kToWorking[p.in_space][i];
        slots.from_working[i] = kFromWorking[p.out_space][i];
    }
    slots.grain_mono = luts.grain_mono() ? 1 : 0;
    slots.match_tex = (p.reserved2[0] & LutSet::kMatchOn) ? luts.texture(LutSet::kMatchSlot) : 0;
    slots.match_size = p.reserved;
    if (const float* t = luts.grain_table()) {
        for (int i = 0; i < 3 * kGrainTableSize; ++i) slots.grain[i] = t[i];
    } else {
        for (int i = 0; i < kGrainTableSize; ++i) {
            double u = (kGrainTableLo + (kGrainTableHi - kGrainTableLo) * i / (kGrainTableSize - 1.0)) / kGrainTonalWidth;
            float v = static_cast<float>(kGrainTonalFloor + (1.0 - kGrainTonalFloor) * std::exp(-0.5 * u * u));
            for (int c = 0; c < 3; ++c) slots.grain[c * kGrainTableSize + i] = v;
        }
    }
    return slots;
}

__device__ float3 to_lin(float4 px, const RenderParams& p, const SlotSet& slots) {
    float3 rgb = make_float3(sanitize(px.x), sanitize(px.y), sanitize(px.z));
    if (p.premultiplied && px.w > 0.0f && isfinite(px.w)) rgb = make_float3(rgb.x / px.w, rgb.y / px.w, rgb.z / px.w);
    float3 dec = make_float3(sanitize(decode(p.in_space, rgb.x)), sanitize(decode(p.in_space, rgb.y)),
                             sanitize(decode(p.in_space, rgb.z)));
    float3 w = mul(slots.to_working, dec);
    return make_float3(sanitize(w.x), sanitize(w.y), sanitize(w.z));
}

__device__ __forceinline__ float4 src_px(Image src, int gx, int gy) {
    return reinterpret_cast<const float4*>(src.base + static_cast<ptrdiff_t>(gy - src.bounds.y1) * src.pitch)[gx - src.bounds.x1];
}

__global__ void region_prepare(Image src, Rect e, float4* lin, RenderParams p, SlotSet slots) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    int w = e.x2 - e.x1;
    if (x >= w || y >= e.y2 - e.y1) return;
    float3 v = to_lin(src_px(src, e.x1 + x, e.y1 + y), p, slots);
    float g = exp2f(p.exposure_ev);
    lin[y * w + x] = make_float4(v.x * g, v.y * g, v.z * g, 0.0f);
}

__global__ void blur_h_clamp(const float4* in, float4* out, int w, int h, const float* k, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = -r; i <= r; ++i) {
        float4 v = in[y * w + min(max(x + i, 0), w - 1)];
        float kk = k[i + r];
        a.x = fmaf(kk, v.x, a.x);
        a.y = fmaf(kk, v.y, a.y);
        a.z = fmaf(kk, v.z, a.z);
    }
    out[y * w + x] = a;
}

__global__ void blur_v_clamp(const float4* in, float4* out, int w, int h, const float* k, int r) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i = -r; i <= r; ++i) {
        float4 v = in[min(max(y + i, 0), h - 1) * w + x];
        float kk = k[i + r];
        a.x = fmaf(kk, v.x, a.x);
        a.y = fmaf(kk, v.y, a.y);
        a.z = fmaf(kk, v.z, a.z);
    }
    out[y * w + x] = a;
}

__global__ void halation_combine(float4* lin, const float4* narrow, const float4* wide, int n, float amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 v = lin[i], a = narrow[i], b = wide[i];
    float m = kHalationMix;
    lin[i] = make_float4(v.x + amount * 1.0f * (m * a.x + (1.0f - m) * b.x - v.x),
                         v.y + amount * 0.25f * (m * a.y + (1.0f - m) * b.y - v.y), v.z, 0.0f);
}

__global__ void bloom_combine(float4* lin, const float4* blurred, int n, float amount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 v = lin[i], b = blurred[i];
    lin[i] = make_float4(v.x + amount * (b.x - v.x), v.y + amount * (b.y - v.y), v.z + amount * (b.z - v.z), 0.0f);
}

__global__ void downsample2(const float4* in, int w, int h, float4* out, int w2, int h2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w2 || y >= h2) return;
    int x0 = min(2 * x, w - 1), x1 = min(2 * x + 1, w - 1), y0 = min(2 * y, h - 1), y1 = min(2 * y + 1, h - 1);
    float4 a = in[y0 * w + x0], b = in[y0 * w + x1], c = in[y1 * w + x0], d = in[y1 * w + x1];
    out[y * w2 + x] = make_float4(0.25f * (a.x + b.x + c.x + d.x), 0.25f * (a.y + b.y + c.y + d.y),
                                  0.25f * (a.z + b.z + c.z + d.z), 0.0f);
}

__device__ float4 sample_half(const float4* hb, int w2, int h2, int x, int y) {
    float fx = fminf(fmaxf((x + 0.5f) * 0.5f - 0.5f, 0.0f), w2 - 1.0f);
    float fy = fminf(fmaxf((y + 0.5f) * 0.5f - 0.5f, 0.0f), h2 - 1.0f);
    int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
    int x1 = min(x0 + 1, w2 - 1), y1 = min(y0 + 1, h2 - 1);
    float ax = fx - x0, ay = fy - y0;
    float4 a = hb[y0 * w2 + x0], b = hb[y0 * w2 + x1], c = hb[y1 * w2 + x0], d = hb[y1 * w2 + x1];
    float4 top = make_float4(a.x + ax * (b.x - a.x), a.y + ax * (b.y - a.y), a.z + ax * (b.z - a.z), 0.0f);
    float4 bot = make_float4(c.x + ax * (d.x - c.x), c.y + ax * (d.y - c.y), c.z + ax * (d.z - c.z), 0.0f);
    return make_float4(top.x + ay * (bot.x - top.x), top.y + ay * (bot.y - top.y), top.z + ay * (bot.z - top.z), 0.0f);
}

__global__ void halation_combine_half(float4* lin, const float4* narrow, const float4* wide, int w, int h, int w2, int h2,
                                      float amount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    float4 v = lin[i], a = sample_half(narrow, w2, h2, x, y), b = sample_half(wide, w2, h2, x, y);
    float m = kHalationMix;
    lin[i] = make_float4(v.x + amount * 1.0f * (m * a.x + (1.0f - m) * b.x - v.x),
                         v.y + amount * 0.25f * (m * a.y + (1.0f - m) * b.y - v.y), v.z, 0.0f);
}

__global__ void bloom_combine_half(float4* lin, const float4* blurred, int w, int h, int w2, int h2, float amount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = y * w + x;
    float4 v = lin[i], b = sample_half(blurred, w2, h2, x, y);
    lin[i] = make_float4(v.x + amount * (b.x - v.x), v.y + amount * (b.y - v.y), v.z + amount * (b.z - v.z), 0.0f);
}

__global__ void region_film(float4* lin, int ox, int oy, int w, int h, RenderParams p, SlotSet slots) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float4 v = lin[y * w + x];
    float3 l = make_float3(v.x, v.y, v.z);
    float4 nz;
    if (p.grain_amount > 0.0f) nz = grain_noise(p, ox + x, oy + y, slots.grain_mono);
    float3 film = film_stage(l, p, slots, p.grain_amount > 0.0f ? &nz : nullptr);
    lin[y * w + x] = make_float4(film.x, film.y, film.z, 0.0f);
}

__global__ void region_final(Image src, Rect e, Rect win, const float4* film, char* dst, size_t dst_pitch, Rect dst_bounds,
                             RenderParams p, SlotSet slots) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= win.x2 - win.x1 || y >= win.y2 - win.y1) return;
    int gx = win.x1 + x, gy = win.y1 + y;
    float4 px = src_px(src, gx, gy);
    float3 w = to_lin(px, p, slots);
    float3 r = mul(slots.from_working, w);
    float3 ref = make_float3(sanitize(encode(p.out_space, r.x)), sanitize(encode(p.out_space, r.y)),
                             sanitize(encode(p.out_space, r.z)));
    float4 f = film[(gy - e.y1) * (e.x2 - e.x1) + (gx - e.x1)];
    float3 fo = mul(slots.from_working, make_float3(f.x, f.y, f.z));
    fo = make_float3(encode(p.out_space, fo.x), encode(p.out_space, fo.y), encode(p.out_space, fo.z));
    float4 out = make_float4(ref.x + p.impact * (fo.x - ref.x), ref.y + p.impact * (fo.y - ref.y),
                             ref.z + p.impact * (fo.z - ref.z), px.w);
    if (p.premultiplied) out = make_float4(out.x * premult_alpha(px.w), out.y * premult_alpha(px.w), out.z * premult_alpha(px.w), px.w);
    reinterpret_cast<float4*>(dst + static_cast<ptrdiff_t>(gy - dst_bounds.y1) * dst_pitch)[gx - dst_bounds.x1] = out;
}

bool match_bad(const RenderParams& p, const LutSet& luts) {
    return (p.reserved2[0] & LutSet::kMatchOn) && (!luts.texture(LutSet::kMatchSlot) || p.reserved < 2 || p.reserved != luts.match_size());
}

int blur_margin(float sigma) { return static_cast<int>(ceilf(3.0f * fminf(fmaxf(sigma, 0.5f), kSpatialMaxSigmaPx))); }

cudaError_t blur(Pool& pool, const float4* in, float4* out, float4* tmp, int w, int h, float sigma) {
    sigma = fminf(fmaxf(sigma, 0.5f), kSpatialMaxSigmaPx);
    int r = 0;
    std::vector<float> k = gauss_weights(sigma, r, nullptr);
    float* weights = nullptr;
    cudaError_t e = upload_weights(pool, k, &weights);
    if (e != cudaSuccess) return e;
    dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
    blur_h_clamp<<<grid, block, 0, pool.stream>>>(in, tmp, w, h, weights, r);
    blur_v_clamp<<<grid, block, 0, pool.stream>>>(tmp, out, w, h, weights, r);
    return cudaGetLastError();
}

}

LutSet::~LutSet() {
    if (match_borrowed_) tex_[kMatchSlot] = 0;
    for (int i = 0; i < 5; ++i) {
        if (tex_[i]) cudaDestroyTextureObject(tex_[i]);
        if (arr_[i]) cudaFreeArray(arr_[i]);
    }
}

bool LutSet::ready() const { return size_ > 1 && tex_[0] && tex_[1] && tex_[2]; }

cudaError_t LutSet::upload(int slot, const float* rgb, int size) {
    if (slot < 0 || slot > kMatchSlot || size < 2 || (slot != kMatchSlot && size_ && size_ != size)) return cudaErrorInvalidValue;
    size_t count = static_cast<size_t>(size) * size * size;
    std::vector<float4> rgba(count);
    for (size_t i = 0; i < count; ++i) rgba[i] = make_float4(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2], 0.0f);
    if (slot == kMatchSlot && match_borrowed_) tex_[slot] = 0, match_borrowed_ = false;
    if (tex_[slot]) cudaDestroyTextureObject(tex_[slot]), tex_[slot] = 0;
    if (arr_[slot]) cudaFreeArray(arr_[slot]), arr_[slot] = nullptr;
    if (slot == kMatchSlot) match_size_ = 0;
    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float4>();
    cudaExtent ext = make_cudaExtent(size, size, size);
    cudaError_t e = cudaMalloc3DArray(&arr_[slot], &fmt, ext);
    if (e != cudaSuccess) return e;
    cudaMemcpy3DParms cp = {};
    cp.srcPtr = make_cudaPitchedPtr(rgba.data(), size * sizeof(float4), size, size);
    cp.dstArray = arr_[slot];
    cp.extent = ext;
    cp.kind = cudaMemcpyHostToDevice;
    if ((e = cudaMemcpy3D(&cp)) != cudaSuccess) return e;
    cudaResourceDesc res = {};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = arr_[slot];
    cudaTextureDesc td = {};
    td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModePoint;
    td.readMode = cudaReadModeElementType;
    td.normalizedCoords = 0;
    if ((e = cudaCreateTextureObject(&tex_[slot], &res, &td, nullptr)) != cudaSuccess) return e;
    (slot == kMatchSlot ? match_size_ : size_) = size;
    return cudaSuccess;
}

cudaError_t render(const float* src, size_t src_pitch, float* dst, size_t dst_pitch, int width, int height,
                   const RenderParams& params, const LutSet& luts, cudaStream_t stream) {
    if (width <= 0 || height <= 0 || params.in_space < 0 || params.in_space >= kSpaceCount ||
        params.out_space < 0 || params.out_space >= kSpaceCount)
        return cudaErrorInvalidValue;
    if (params.impact == 0.0f)
        return cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width * sizeof(float4), height,
                                 cudaMemcpyDeviceToDevice, stream);
    if (!luts.ready()) return cudaErrorNotReady;
    if (match_bad(params, luts)) return cudaErrorInvalidValue;
    RenderParams p = params;
    p.lut_size = luts.size();
    SlotSet slots = make_slots(p, luts);
    dim3 block(16, 16), grid((width + 15) / 16, (height + 15) / 16);
    film_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const char*>(src), src_pitch,
                                            reinterpret_cast<char*>(dst), dst_pitch, width, height, p, slots);
    return cudaGetLastError();
}

cudaError_t render_region(Image src, float* dst_base, size_t dst_pitch, Rect dst_bounds, Rect win, const RenderParams& params,
                          const LutSet& luts, cudaStream_t stream) {
    win.x1 = std::max({win.x1, src.bounds.x1, dst_bounds.x1});
    win.y1 = std::max({win.y1, src.bounds.y1, dst_bounds.y1});
    win.x2 = std::min({win.x2, src.bounds.x2, dst_bounds.x2});
    win.y2 = std::min({win.y2, src.bounds.y2, dst_bounds.y2});
    if (win.width() <= 0 || win.height() <= 0) return cudaSuccess;
    bool spatial = params.impact != 0.0f && (params.halation_amount > 0.0f || params.bloom_amount > 0.0f);
    if (!spatial) {
        RenderParams p = params;
        p.origin_x = win.x1;
        p.origin_y = win.y1;
        const char* s = src.base + static_cast<ptrdiff_t>(win.y1 - src.bounds.y1) * src.pitch + (win.x1 - src.bounds.x1) * 16;
        char* d = reinterpret_cast<char*>(dst_base) + static_cast<ptrdiff_t>(win.y1 - dst_bounds.y1) * dst_pitch +
                  (win.x1 - dst_bounds.x1) * 16;
        return render(reinterpret_cast<const float*>(s), src.pitch, reinterpret_cast<float*>(d), dst_pitch, win.width(),
                      win.height(), p, luts, stream);
    }
    if (params.in_space < 0 || params.in_space >= kSpaceCount || params.out_space < 0 || params.out_space >= kSpaceCount)
        return cudaErrorInvalidValue;
    if (!luts.ready()) return cudaErrorNotReady;
    if (match_bad(params, luts)) return cudaErrorInvalidValue;
    RenderParams p = params;
    p.lut_size = luts.size();
    SlotSet slots = make_slots(p, luts);
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
    Pool pool(stream);
    float4 *lin = nullptr, *t1 = nullptr, *t2 = nullptr, *tmp = nullptr, *lh = nullptr;
    cudaError_t err;
    if ((err = pool.alloc(&lin, n)) != cudaSuccess || (err = pool.alloc(&t1, nb)) != cudaSuccess ||
        (err = pool.alloc(&t2, nb)) != cudaSuccess || (err = pool.alloc(&tmp, nb)) != cudaSuccess ||
        (half && (err = pool.alloc(&lh, nb)) != cudaSuccess))
        return err;
    dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16), hgrid((w2 + 15) / 16, (h2 + 15) / 16);
    int lin_blocks = static_cast<int>((n + 255) / 256);
    region_prepare<<<grid, block, 0, stream>>>(src, e, lin, p, slots);
    if (p.halation_amount > 0.0f) {
        if (half) {
            downsample2<<<hgrid, block, 0, stream>>>(lin, w, h, lh, w2, h2);
            if ((err = blur(pool, lh, t1, tmp, w2, h2, 0.5f * kHalationNarrow * p.halation_radius_px)) != cudaSuccess) return err;
            if ((err = blur(pool, lh, t2, tmp, w2, h2, 0.5f * p.halation_radius_px)) != cudaSuccess) return err;
            halation_combine_half<<<grid, block, 0, stream>>>(lin, t1, t2, w, h, w2, h2, p.halation_amount);
        } else {
            if ((err = blur(pool, lin, t1, tmp, w, h, kHalationNarrow * p.halation_radius_px)) != cudaSuccess) return err;
            if ((err = blur(pool, lin, t2, tmp, w, h, p.halation_radius_px)) != cudaSuccess) return err;
            halation_combine<<<lin_blocks, 256, 0, stream>>>(lin, t1, t2, static_cast<int>(n), p.halation_amount);
        }
    }
    region_film<<<grid, block, 0, stream>>>(lin, e.x1, e.y1, w, h, p, slots);
    if (p.bloom_amount > 0.0f) {
        if (half) {
            downsample2<<<hgrid, block, 0, stream>>>(lin, w, h, lh, w2, h2);
            if ((err = blur(pool, lh, t1, tmp, w2, h2, 0.5f * p.bloom_radius_px)) != cudaSuccess) return err;
            bloom_combine_half<<<grid, block, 0, stream>>>(lin, t1, w, h, w2, h2, p.bloom_amount);
        } else {
            if ((err = blur(pool, lin, t1, tmp, w, h, p.bloom_radius_px)) != cudaSuccess) return err;
            bloom_combine<<<lin_blocks, 256, 0, stream>>>(lin, t1, static_cast<int>(n), p.bloom_amount);
        }
    }
    dim3 wgrid((win.width() + 15) / 16, (win.height() + 15) / 16);
    region_final<<<wgrid, block, 0, stream>>>(src, e, win, lin, reinterpret_cast<char*>(dst_base), dst_pitch, dst_bounds, p,
                                             slots);
    return cudaGetLastError();
}

}
