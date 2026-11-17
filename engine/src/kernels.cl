typedef struct {
    int in_space;
    int out_space;
    float exposure_ev;
    float push_pull;
    float impact;
    int premultiplied;
    int lut_size;
    int reserved;
    float grain_amount;
    float grain_sigma_px;
    uint grain_seed;
    int frame;
    int origin_x;
    int origin_y;
    float halation_amount;
    float halation_radius_px;
    float bloom_amount;
    float bloom_radius_px;
    float print_log_offset[3];
    int spatial_half;
    int reserved2[1];
    float match_protect;
} RenderParams;

typedef struct {
    RenderParams p;
    float weight[3];
    float to_working[9];
    float from_working[9];
    int two_stage;
    float print_lo;
    float print_inv;
    float grain[3 * kGrainTableSize];
    int grain_mono;
    int match_on;
    int match_size;
} Args;

typedef __constant const Args* A_t;
typedef __global const float4* L_t;

#define kDiA 0.0075f
#define kDiB 7.0f
#define kDiC 0.07329248f
#define kDiM 10.44426855f
#define kDiLinCut 0.00262409f
#define kDiLogCut 0.02740668f
#define kCctLinCut 0.0078125f
#define kCctLogCut 0.155251141552511f
#define kCctA 10.5402377416545f
#define kCctB 0.0729055341958355f
#define kFMax 65504.0f

static inline float sanitize(float x) {
    if (isnan(x)) return 0.0f;
    return fmin(fmax(x, -kFMax), kFMax);
}

static inline float di_encode(float x) { return x <= kDiLinCut ? kDiM * x : kDiC * (log2(x + kDiA) + kDiB); }
static inline float di_decode(float y) { return y <= kDiLogCut ? y / kDiM : exp2(y / kDiC - kDiB) - kDiA; }
static inline float cct_encode(float x) { return x <= kCctLinCut ? kCctA * x + kCctB : (log2(x) + 9.72f) / 17.52f; }
static inline float cct_decode(float y) { return y <= kCctLogCut ? (y - kCctB) / kCctA : exp2(y * 17.52f - 9.72f); }
static inline float g24_encode(float x) { return copysign(pow(fabs(x), 1.0f / 2.4f), x); }
static inline float g24_decode(float y) { return copysign(pow(fabs(y), 2.4f), y); }

static inline float encode(int space, float x) {
    switch (space) {
        case kSpace_dwg_di: return di_encode(x);
        case kSpace_acescct: return cct_encode(x);
        case kSpace_rec709_g24: return g24_encode(x);
        default: return x;
    }
}

static inline float decode(int space, float y) {
    switch (space) {
        case kSpace_dwg_di: return di_decode(y);
        case kSpace_acescct: return cct_decode(y);
        case kSpace_rec709_g24: return g24_decode(y);
        default: return y;
    }
}

static inline float3 mul(__constant const float* m, float3 v) {
    return (float3)(m[0] * v.x + m[1] * v.y + m[2] * v.z, m[3] * v.x + m[4] * v.y + m[5] * v.z,
                    m[6] * v.x + m[7] * v.y + m[8] * v.z);
}

static inline float4 texel(L_t tex, int n, int r, int g, int b) { return tex[(b * n + g) * n + r]; }

static inline float4 lerp4(float4 a, float4 b, float t) {
    return (float4)(fma(t, b.x - a.x, a.x), fma(t, b.y - a.y, a.y), fma(t, b.z - a.z, a.z), 0.0f);
}

float3 sample_raw(L_t tex, float3 c, int n) {
    float pr = fmin(fmax(c.x, 0.0f), 1.0f) * (n - 1);
    float pg = fmin(fmax(c.y, 0.0f), 1.0f) * (n - 1);
    float pb = fmin(fmax(c.z, 0.0f), 1.0f) * (n - 1);
    int r = min((int)pr, n - 2), g = min((int)pg, n - 2), b = min((int)pb, n - 2);
    float fr = pr - r, fg = pg - g, fb = pb - b;
    float4 c0 = lerp4(lerp4(texel(tex, n, r, g, b), texel(tex, n, r + 1, g, b), fr),
                      lerp4(texel(tex, n, r, g + 1, b), texel(tex, n, r + 1, g + 1, b), fr), fg);
    float4 c1 = lerp4(lerp4(texel(tex, n, r, g, b + 1), texel(tex, n, r + 1, g, b + 1), fr),
                      lerp4(texel(tex, n, r, g + 1, b + 1), texel(tex, n, r + 1, g + 1, b + 1), fr), fg);
    float4 v = lerp4(c0, c1, fb);
    return (float3)(v.x, v.y, v.z);
}

float3 sample_match(L_t tex, float3 lin, int n) {
    float encoded[3] = {di_encode(lin.x), di_encode(lin.y), di_encode(lin.z)};
    int corner[3], order[3] = {0, 1, 2};
    float fraction[3];
    for (int c = 0; c < 3; ++c) {
        float position = fmin(fmax(encoded[c], 0.0f), 1.0f) * (n - 1);
        corner[c] = min((int)position, n - 2);
        fraction[c] = position - corner[c];
    }
    for (int i = 0; i < 2; ++i)
        for (int j = i + 1; j < 3; ++j)
            if (fraction[order[j]] > fraction[order[i]]) {
                int swap = order[i]; order[i] = order[j]; order[j] = swap;
            }
    float4 previous = texel(tex, n, corner[0], corner[1], corner[2]);
    float3 result = (float3)(previous.x, previous.y, previous.z);
    for (int i = 0; i < 3; ++i) {
        ++corner[order[i]];
        float4 next = texel(tex, n, corner[0], corner[1], corner[2]);
        float weight = fraction[order[i]];
        result.x += weight * (next.x - previous.x);
        result.y += weight * (next.y - previous.y);
        result.z += weight * (next.z - previous.z);
        previous = next;
    }
    return (float3)(di_decode(result.x), di_decode(result.y), di_decode(result.z));
}

float3 sample_slot(L_t tex, float3 lin, int n) {
    float3 v = sample_raw(tex, (float3)(di_encode(lin.x), di_encode(lin.y), di_encode(lin.z)), n);
    return (float3)(di_decode(v.x), di_decode(v.y), di_decode(v.z));
}

static inline float grain_lookup(__constant const float* t, float v) {
    float u = log10(fmax(v, 9.5367432e-7f) / 0.18f);
    float f = (fmin(fmax(u, kGrainTableLo), kGrainTableHi) - kGrainTableLo) *
              ((kGrainTableSize - 1) / (kGrainTableHi - kGrainTableLo));
    int i = min((int)f, kGrainTableSize - 2);
    return fma(f - i, t[i + 1] - t[i], t[i]);
}

static inline float3 grain_offset(float3 scene, float4 n, float amount, A_t a) {
    float k = amount * kGrainBaseSigmaLog10;
    if (a->grain_mono) {
        float s = k * grain_lookup(a->grain + kGrainTableSize, scene.y) * n.x;
        return (float3)(s, s, s);
    }
    return (float3)(k * grain_lookup(a->grain, scene.x) * n.x, k * grain_lookup(a->grain + kGrainTableSize, scene.y) * n.y,
                    k * grain_lookup(a->grain + 2 * kGrainTableSize, scene.z) * n.z);
}

float3 film_eval(float3 lin, A_t a, L_t t0, L_t t1, L_t t2, L_t tp, int has_grain, float4 n) {
    float3 acc = (float3)(0.0f, 0.0f, 0.0f);
    float3 c = (float3)(di_encode(lin.x), di_encode(lin.y), di_encode(lin.z));
    int size = a->p.lut_size;
    for (int i = 0; i < 3; ++i) {
        if (a->weight[i] == 0.0f) continue;
        L_t tex = i == 0 ? t0 : i == 1 ? t1 : t2;
        float3 s = a->two_stage ? sample_raw(tex, c, size) : sample_slot(tex, lin, size);
        acc.x += a->weight[i] * s.x;
        acc.y += a->weight[i] * s.y;
        acc.z += a->weight[i] * s.z;
    }
    if (has_grain) {
        float3 g = grain_offset(lin, n, a->p.grain_amount, a);
        if (!a->two_stage) return (float3)(acc.x * exp10(-g.x), acc.y * exp10(-g.y), acc.z * exp10(-g.z));
        acc = (float3)(acc.x - g.x, acc.y - g.y, acc.z - g.z);
    }
    if (!a->two_stage) return acc;
    float3 q = (float3)((acc.x + a->p.print_log_offset[0] - a->print_lo) * a->print_inv,
                        (acc.y + a->p.print_log_offset[1] - a->print_lo) * a->print_inv,
                        (acc.z + a->p.print_log_offset[2] - a->print_lo) * a->print_inv);
    float3 v = sample_raw(tp, q, size);
    return (float3)(di_decode(v.x), di_decode(v.y), di_decode(v.z));
}

static inline float skin_smooth(float e0, float e1, float x) {
    float t = fmin(fmax((x - e0) / (e1 - e0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static inline float lab_f(float t) { return t > 0.008856451679f ? cbrt(t) : 7.787037037f * t + 0.137931034f; }

float skin_weight(float3 c) {
    float r = 1.8986149f * c.x - 0.79217618f * c.y - 0.10643872f * c.z;
    float g = -0.16894879f * c.x + 1.48897575f * c.y - 0.32002697f * c.z;
    float b = -0.12153916f * c.x - 0.31567585f * c.y + 1.43721501f * c.z;
    float fx = lab_f((0.41239080f * r + 0.35758434f * g + 0.18048079f * b) / 0.95045593f);
    float fy = lab_f(0.21263901f * r + 0.71516868f * g + 0.07219232f * b);
    float fz = lab_f((0.01933082f * r + 0.11919478f * g + 0.95053215f * b) / 1.08905775f);
    float L = 116.0f * fy - 16.0f, A = 500.0f * (fx - fy), B = 200.0f * (fy - fz);
    if (!(L >= 15.0f && L <= 95.0f)) return 0.0f;
    float C = hypot(A, B), h = atan2(B, A) * 57.29577951f;
    if (h < 0.0f) h += 360.0f;
    return skin_smooth(10.0f, 20.0f, h) * (1.0f - skin_smooth(75.0f, 85.0f, h)) * skin_smooth(2.0f, 8.0f, C) *
           (1.0f - skin_smooth(45.0f, 51.0f, C));
}

float3 film_stage(float3 lin, A_t a, L_t t0, L_t t1, L_t t2, L_t tp, L_t tm, int has_grain, float4 n) {
    float3 film = (a->p.reserved2[0] & FE_FILM_BYPASS) ? lin : film_eval(lin, a, t0, t1, t2, tp, has_grain, n);
    if (!a->match_on) return film;
    float3 m = sample_match(tm, film, a->match_size);
    if (!(a->p.match_protect > 0.0f)) return m;
    float w = 1.0f - fmin(a->p.match_protect, 1.0f) * skin_weight(film);
    return (float3)(film.x + w * (m.x - film.x), film.y + w * (m.y - film.y), film.z + w * (m.z - film.z));
}

static inline uint hash32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__constant float kGrainLayerScale[3] = {1.0f, 0.9f, 1.2f};
__constant float kGrainCloudCdf[4] = {0.36787944f, 0.73575888f, 0.91969860f, 0.98101184f};

static inline float unit24(uint h) { return ((h >> 8) + 0.5f) * 5.9604645e-8f; }

float clouds(uint base, float px, float py, float radius) {
    float ir = 1.0f / radius, qx = px * ir, qy = py * ir;
    int cx0 = (int)floor(qx), cy0 = (int)floor(qy);
    float lx = qx - cx0, ly = qy - cy0, acc = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            uint h = hash32(base ^ ((uint)(cx0 + dx) * 0x8DA6B343u) ^ ((uint)(cy0 + dy) * 0xD8163841u));
            float u = unit24(h);
            int count = (u > kGrainCloudCdf[0]) + (u > kGrainCloudCdf[1]) + (u > kGrainCloudCdf[2]) + (u > kGrainCloudCdf[3]);
            for (int j = 0; j < count; ++j) {
                h = hash32(h + 0x9E3779B9u);
                float ex = lx - (dx + ((h & 2047u) + 0.5f) * (1.0f / 2048.0f));
                float ey = ly - (dy + (((h >> 11) & 2047u) + 0.5f) * (1.0f / 2048.0f));
                float a = ((h >> 22) + 0.5f) * (2.0f / 1024.0f);
                float b = fmax(1.0f - (ex * ex + ey * ey), 0.0f);
                acc = fma(a, b * b, acc);
            }
        }
    }
    return (acc - kGrainCloudMean) * (1.0f / kGrainCloudStd);
}

float4 grain_noise(A_t a, int gx, int gy) {
    float sigma = fmin(fmax(a->p.grain_sigma_px, kGrainMinBlurPx), kGrainMaxBlurPx);
    float px = gx + 0.5f, py = gy + 0.5f, r = kGrainCloudRadius * sigma;
    uint b = hash32(hash32(a->p.grain_seed) ^ (uint)a->p.frame);
    float n0 = clouds(hash32(b ^ 0u), px, py, r * kGrainLayerScale[0]);
    float n1 = a->grain_mono ? n0 : clouds(hash32(b ^ 1u), px, py, r * kGrainLayerScale[1]);
    float n2 = a->grain_mono ? n0 : clouds(hash32(b ^ 2u), px, py, r * kGrainLayerScale[2]);
    return (float4)(n0, n1, n2, 0.0f);
}

static inline float premult_alpha(float a) { return isfinite(a) ? a : (a > 0.0f ? 1.0f : 0.0f); }

float3 to_lin(float4 px, A_t a) {
    float3 rgb = (float3)(sanitize(px.x), sanitize(px.y), sanitize(px.z));
    if (a->p.premultiplied && px.w > 0.0f && isfinite(px.w)) rgb = (float3)(rgb.x / px.w, rgb.y / px.w, rgb.z / px.w);
    float3 dec = (float3)(sanitize(decode(a->p.in_space, rgb.x)), sanitize(decode(a->p.in_space, rgb.y)),
                          sanitize(decode(a->p.in_space, rgb.z)));
    float3 w = mul(a->to_working, dec);
    return (float3)(sanitize(w.x), sanitize(w.y), sanitize(w.z));
}

float4 finish(float4 px, float3 w, float3 film, A_t a) {
    float3 r = mul(a->from_working, w);
    float3 ref = (float3)(sanitize(encode(a->p.out_space, r.x)), sanitize(encode(a->p.out_space, r.y)),
                          sanitize(encode(a->p.out_space, r.z)));
    float3 fo = mul(a->from_working, film);
    fo = (float3)(encode(a->p.out_space, fo.x), encode(a->p.out_space, fo.y), encode(a->p.out_space, fo.z));
    float impact = a->p.impact;
    float4 out = (float4)(ref.x + impact * (fo.x - ref.x), ref.y + impact * (fo.y - ref.y), ref.z + impact * (fo.z - ref.z), px.w);
    if (a->p.premultiplied)
        out = (float4)(out.x * premult_alpha(px.w), out.y * premult_alpha(px.w), out.z * premult_alpha(px.w), px.w);
    return out;
}

__kernel void args_size(__global int* out) { out[0] = (int)sizeof(Args); }

__kernel void film_kernel(__global const float* src, ulong src_off, ulong src_pitch, __global float* dst, ulong dst_off,
                          ulong dst_pitch, int width, int height, A_t a, L_t t0, L_t t1, L_t t2, L_t tp, L_t tm) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= width || y >= height) return;
    float4 px = vload4(0, src + src_off + y * src_pitch + 4 * x);
    float3 w = to_lin(px, a);
    float g = exp2(a->p.exposure_ev);
    float3 lin = (float3)(w.x * g, w.y * g, w.z * g);
    int has_grain = a->p.grain_amount > 0.0f;
    float4 nz = has_grain ? grain_noise(a, a->p.origin_x + x, a->p.origin_y + y) : (float4)(0.0f);
    float3 film = film_stage(lin, a, t0, t1, t2, tp, tm, has_grain, nz);
    vstore4(finish(px, w, film, a), 0, dst + dst_off + y * dst_pitch + 4 * x);
}

static inline float4 src_px(__global const float* src, ulong off, ulong pitch, int bx, int by, int gx, int gy) {
    return vload4(0, src + off + (long)(gy - by) * pitch + 4 * (gx - bx));
}

__kernel void region_prepare(__global const float* src, ulong off, ulong pitch, int bx, int by, int ex, int ey, int w, int h,
                             __global float4* lin, A_t a) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float3 v = to_lin(src_px(src, off, pitch, bx, by, ex + x, ey + y), a);
    float g = exp2(a->p.exposure_ev);
    lin[y * w + x] = (float4)(v.x * g, v.y * g, v.z * g, 0.0f);
}

__kernel void blur_h_clamp(__global const float4* in, __global float4* out, int w, int h, __global const float* k, int r) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float4 a = (float4)(0.0f);
    for (int i = -r; i <= r; ++i) {
        float4 v = in[y * w + min(max(x + i, 0), w - 1)];
        float kk = k[i + r];
        a.x = fma(kk, v.x, a.x);
        a.y = fma(kk, v.y, a.y);
        a.z = fma(kk, v.z, a.z);
    }
    out[y * w + x] = a;
}

__kernel void blur_v_clamp(__global const float4* in, __global float4* out, int w, int h, __global const float* k, int r) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float4 a = (float4)(0.0f);
    for (int i = -r; i <= r; ++i) {
        float4 v = in[min(max(y + i, 0), h - 1) * w + x];
        float kk = k[i + r];
        a.x = fma(kk, v.x, a.x);
        a.y = fma(kk, v.y, a.y);
        a.z = fma(kk, v.z, a.z);
    }
    out[y * w + x] = a;
}

__kernel void halation_combine(__global float4* lin, __global const float4* narrow, __global const float4* wide, int n, float amount) {
    int i = get_global_id(0);
    if (i >= n) return;
    float4 v = lin[i], a = narrow[i], b = wide[i];
    float m = kHalationMix;
    lin[i] = (float4)(v.x + amount * 1.0f * (m * a.x + (1.0f - m) * b.x - v.x),
                      v.y + amount * 0.25f * (m * a.y + (1.0f - m) * b.y - v.y), v.z, 0.0f);
}

__kernel void bloom_combine(__global float4* lin, __global const float4* blurred, int n, float amount) {
    int i = get_global_id(0);
    if (i >= n) return;
    float4 v = lin[i], b = blurred[i];
    lin[i] = (float4)(v.x + amount * (b.x - v.x), v.y + amount * (b.y - v.y), v.z + amount * (b.z - v.z), 0.0f);
}

__kernel void downsample2(__global const float4* in, int w, int h, __global float4* out, int w2, int h2) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w2 || y >= h2) return;
    int x0 = min(2 * x, w - 1), x1 = min(2 * x + 1, w - 1), y0 = min(2 * y, h - 1), y1 = min(2 * y + 1, h - 1);
    float4 a = in[y0 * w + x0], b = in[y0 * w + x1], c = in[y1 * w + x0], d = in[y1 * w + x1];
    out[y * w2 + x] = (float4)(0.25f * (a.x + b.x + c.x + d.x), 0.25f * (a.y + b.y + c.y + d.y),
                               0.25f * (a.z + b.z + c.z + d.z), 0.0f);
}

float4 sample_half(__global const float4* hb, int w2, int h2, int x, int y) {
    float fx = fmin(fmax((x + 0.5f) * 0.5f - 0.5f, 0.0f), w2 - 1.0f);
    float fy = fmin(fmax((y + 0.5f) * 0.5f - 0.5f, 0.0f), h2 - 1.0f);
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = min(x0 + 1, w2 - 1), y1 = min(y0 + 1, h2 - 1);
    float ax = fx - x0, ay = fy - y0;
    float4 a = hb[y0 * w2 + x0], b = hb[y0 * w2 + x1], c = hb[y1 * w2 + x0], d = hb[y1 * w2 + x1];
    float4 top = (float4)(a.x + ax * (b.x - a.x), a.y + ax * (b.y - a.y), a.z + ax * (b.z - a.z), 0.0f);
    float4 bot = (float4)(c.x + ax * (d.x - c.x), c.y + ax * (d.y - c.y), c.z + ax * (d.z - c.z), 0.0f);
    return (float4)(top.x + ay * (bot.x - top.x), top.y + ay * (bot.y - top.y), top.z + ay * (bot.z - top.z), 0.0f);
}

__kernel void halation_combine_half(__global float4* lin, __global const float4* narrow, __global const float4* wide, int w, int h,
                                    int w2, int h2, float amount) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int i = y * w + x;
    float4 v = lin[i], a = sample_half(narrow, w2, h2, x, y), b = sample_half(wide, w2, h2, x, y);
    float m = kHalationMix;
    lin[i] = (float4)(v.x + amount * 1.0f * (m * a.x + (1.0f - m) * b.x - v.x),
                      v.y + amount * 0.25f * (m * a.y + (1.0f - m) * b.y - v.y), v.z, 0.0f);
}

__kernel void bloom_combine_half(__global float4* lin, __global const float4* blurred, int w, int h, int w2, int h2, float amount) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    int i = y * w + x;
    float4 v = lin[i], b = sample_half(blurred, w2, h2, x, y);
    lin[i] = (float4)(v.x + amount * (b.x - v.x), v.y + amount * (b.y - v.y), v.z + amount * (b.z - v.z), 0.0f);
}

__kernel void region_film(__global float4* lin, int ox, int oy, int w, int h, A_t a, L_t t0, L_t t1, L_t t2, L_t tp, L_t tm) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= w || y >= h) return;
    float4 v = lin[y * w + x];
    int has_grain = a->p.grain_amount > 0.0f;
    float4 nz = has_grain ? grain_noise(a, ox + x, oy + y) : (float4)(0.0f);
    float3 film = film_stage((float3)(v.x, v.y, v.z), a, t0, t1, t2, tp, tm, has_grain, nz);
    lin[y * w + x] = (float4)(film.x, film.y, film.z, 0.0f);
}

__kernel void region_final(__global const float* src, ulong off, ulong pitch, int bx, int by, int ex1, int ey1, int ew, int wx1,
                           int wy1, int ww, int wh, __global const float4* film, __global float* dst, ulong dst_pitch, int dbx,
                           int dby, A_t a) {
    int x = get_global_id(0), y = get_global_id(1);
    if (x >= ww || y >= wh) return;
    int gx = wx1 + x, gy = wy1 + y;
    float4 px = src_px(src, off, pitch, bx, by, gx, gy);
    float4 f = film[(gy - ey1) * ew + (gx - ex1)];
    float4 out = finish(px, to_lin(px, a), (float3)(f.x, f.y, f.z), a);
    vstore4(out, 0, dst + (long)(gy - dby) * dst_pitch + 4 * (gx - dbx));
}
