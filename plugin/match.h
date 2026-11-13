#pragma once
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fxm {

using V3 = std::array<double, 3>;

constexpr double kPcts[7] = {2, 10, 25, 50, 75, 90, 98};
constexpr double kSlopeMin = 0.5, kSlopeMax = 2.0;
constexpr double kSatMin = 0.85, kSatMax = 1.25;
constexpr double kTintMin = 0.93, kTintMax = 1.07;
constexpr double kMinShare = 0.005, kNeutralC = 15.0, kKnotGap = 1e-3;
constexpr double kShadowStops = 4.0;
inline const double kWhiteStops = std::log2(1.0 / 0.18);
constexpr int kLutSize = 33;
constexpr int kAnalysisWidth = 256;
constexpr const char* kCubeHeader = "FIXER reference match only; film, grain, halation, bloom not included";
constexpr double kPcts3[9] = {0.5, 2, 10, 25, 50, 75, 90, 98, 99.5};
constexpr double kSlope3Min = 0.2, kSlope3Max = 5.0;
constexpr double kSat3Min = 0.05, kSat3Max = 2.5;
// Ratios to green; gain_vec normalizes luminance, so these are not exposure gains.
constexpr double kGainMin = 1.0 / 64.0, kGainMax = 64.0;
constexpr double kLiftMax = 4.0, kMerge = 0.5, kLiftEps = 1e-6;
constexpr int kNewtonSteps = 8, kMonoSteps = 20, kMonoGrid = 401;

constexpr double kDwgTo709[9] = {1.8986148993059058,   -0.7921761834040434, -0.10643871590186224,
                                 -0.16894878647615938, 1.488975754118116,   -0.3200269676419564,
                                 -0.12153916060431863, -0.31567585305224316, 1.4372150136565618};
constexpr double k709ToDwg[9] = {0.5627674560071076,  0.3235165887039594, 0.1137159552889327,
                                 0.07775463528504596, 0.749577346163222,  0.17266801855173203,
                                 0.06466919991632825, 0.19199869204629894, 0.743332108037373};
constexpr double k709ToXyz[9] = {0.41239079926595934,  0.35758433938387796, 0.1804807884018343,
                                 0.2126390058715103,   0.7151686787677559,  0.07219231536073371,
                                 0.019330818715591825, 0.11919477979462595, 0.9505321522496606};
constexpr double kWhite[3] = {0.9504559270516716, 1.0, 1.0890577507598784};
constexpr double kDiA = 0.0075, kDiB = 7.0, kDiC = 0.07329248, kDiM = 10.44426855;
constexpr double kDiLinCut = 0.00262409, kDiLogCut = 0.02740668;

inline V3 mul(const double* m, const V3& c) {
    return {m[0] * c[0] + m[1] * c[1] + m[2] * c[2], m[3] * c[0] + m[4] * c[1] + m[5] * c[2],
            m[6] * c[0] + m[7] * c[1] + m[8] * c[2]};
}

inline double luma(const V3& c) { return k709ToXyz[3] * c[0] + k709ToXyz[4] * c[1] + k709ToXyz[5] * c[2]; }

inline double di_encode(double x) { return x <= kDiLinCut ? kDiM * x : kDiC * (std::log2(x + kDiA) + kDiB); }
inline double di_decode(double y) { return y <= kDiLogCut ? y / kDiM : std::exp2(y / kDiC - kDiB) - kDiA; }
inline double spow(double x, double p) { return x < 0.0 ? -std::pow(-x, p) : std::pow(x, p); }
inline double g24_decode(double y) { return spow(y, 2.4); }
inline double g24_encode(double x) { return spow(x, 1.0 / 2.4); }
inline double srgb_decode(double v) { return v <= 12.92 * 0.0031308 ? v / 12.92 : spow((v + 0.055) / 1.055, 2.4); }

inline double lab_f(double t) {
    return t > (24.0 / 116.0) * (24.0 / 116.0) * (24.0 / 116.0) ? spow(t, 1.0 / 3.0) : (841.0 / 108.0) * t + 16.0 / 116.0;
}

inline V3 lab_709(const V3& lin) {
    V3 xyz = mul(k709ToXyz, lin);
    double fx = lab_f(xyz[0] / kWhite[0]), fy = lab_f(xyz[1] / kWhite[1]), fz = lab_f(xyz[2] / kWhite[2]);
    return {116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)};
}

inline double L_of_Y(double y) { return 116.0 * lab_f(y * 100.0 / 100.0) - 16.0; }

inline double Y_of_L(double L) {
    double f = (L + 16.0) / 116.0;
    return (f > 24.0 / 116.0 ? 100.0 * std::pow(f, 3.0) : 100.0 * (f - 16.0 / 116.0) * (108.0 / 841.0)) / 100.0;
}

inline double clip(double v, double lo, double hi) { return std::min(std::max(v, lo), hi); }

inline double sane(double v) { return std::isfinite(v) ? v : 0.0; }

struct Stats {
    int nL = 7;
    double L[9] = {};
    bool has_sat = false;
    double sat = 0.0;
    bool has_tint[3] = {};
    double tint[3][3] = {};
};

struct State {
    int version = 2;
    std::string source = "film";
    int n = 7;
    double curve[9][2] = {};
    double sat = 1.0;
    double tint[3][2] = {{1, 1}, {1, 1}, {1, 1}};
    double zone_L[3] = {};
    Stats ref, frame;
};

constexpr const char* kZones[3] = {"sh", "mid", "hl"};

template <class T>
std::vector<double> downsample(const T* base, std::ptrdiff_t row_stride, int w, int h, int ch, int out_ch, double div,
                               int& out_w, int& out_h, int width = kAnalysisWidth) {
    if (w <= width) {
        out_w = w, out_h = h;
        std::vector<double> out(static_cast<size_t>(w) * h * out_ch);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < out_ch; ++c) out[(static_cast<size_t>(y) * w + x) * out_ch + c] = sane(static_cast<double>(base[y * row_stride + x * ch + c])) / div;
        return out;
    }
    int height = std::max(1, static_cast<int>(std::nearbyint(static_cast<double>(h) * width / w)));
    out_w = width, out_h = height;
    auto spans = [](int n_src, int n_dst, std::vector<int>& first, std::vector<std::vector<double>>& wts) {
        double step = static_cast<double>(n_src) / n_dst;
        first.resize(n_dst);
        wts.resize(n_dst);
        for (int i = 0; i < n_dst; ++i) {
            double lo = i * step, hi = (i + 1) * step;
            int a = std::max(0, static_cast<int>(std::floor(lo))), b = std::min(n_src, static_cast<int>(std::ceil(hi)));
            first[i] = a;
            for (int s = a; s < b; ++s) wts[i].push_back(std::max(std::min(hi, s + 1.0) - std::max(lo, static_cast<double>(s)), 0.0) / step);
        }
    };
    std::vector<int> fy, fx;
    std::vector<std::vector<double>> wy, wx;
    spans(h, height, fy, wy);
    spans(w, width, fx, wx);
    std::vector<double> rows(static_cast<size_t>(w) * out_ch), out(static_cast<size_t>(width) * height * out_ch);
    for (int y = 0; y < height; ++y) {
        std::fill(rows.begin(), rows.end(), 0.0);
        for (size_t k = 0; k < wy[y].size(); ++k) {
            const T* row = base + static_cast<std::ptrdiff_t>(fy[y] + k) * row_stride;
            double wk = wy[y][k];
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < out_ch; ++c) rows[static_cast<size_t>(x) * out_ch + c] += wk * (sane(static_cast<double>(row[x * ch + c])) / div);
        }
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < out_ch; ++c) {
                double acc = 0.0;
                for (size_t k = 0; k < wx[x].size(); ++k) acc += wx[x][k] * rows[(fx[x] + k) * out_ch + c];
                out[(static_cast<size_t>(y) * width + x) * out_ch + c] = acc;
            }
    }
    return out;
}

inline double percentile(const std::vector<double>& sorted, double q) {
    double idx = (sorted.size() - 1) * (q / 100.0);
    size_t i = static_cast<size_t>(std::floor(idx));
    size_t j = std::min(i + 1, sorted.size() - 1);
    double t = idx - i, d = sorted[j] - sorted[i];
    return t >= 0.5 ? sorted[j] - d * (1.0 - t) : sorted[i] + d * t;
}

inline Stats stats_v2(const double* rgb, size_t n, int stride, bool srgb) {
    Stats st;
    std::vector<double> L(n), A(n), B(n), C(n);
    std::vector<char> ok(n);
    for (size_t i = 0; i < n; ++i) {
        V3 enc, lin;
        bool good = true;
        for (int c = 0; c < 3; ++c) {
            enc[c] = clip(sane(rgb[i * stride + c]), 0.0, 1.0);
            good = good && enc[c] < 254.0 / 255.0 && enc[c] > 1.0 / 255.0;
            lin[c] = srgb ? srgb_decode(enc[c]) : g24_decode(enc[c]);
        }
        V3 lab = lab_709(lin);
        L[i] = lab[0], A[i] = lab[1], B[i] = lab[2], C[i] = std::hypot(lab[1], lab[2]);
        ok[i] = good;
    }
    std::vector<double> sorted = L;
    std::sort(sorted.begin(), sorted.end());
    for (int k = 0; k < 7; ++k) st.L[k] = percentile(sorted, kPcts[k]);
    double sh = std::min(st.L[1], 40.0), hl = std::max(st.L[5], 55.0), need = kMinShare * n;
    std::vector<double> mids;
    double sum[3][3] = {}, cnt[3] = {};
    for (size_t i = 0; i < n; ++i) {
        bool mid = L[i] > 25.0 && L[i] < 75.0;
        if (mid && ok[i] && C[i] > 2.0) mids.push_back(C[i]);
        if (!ok[i] || !(C[i] < kNeutralC)) continue;
        bool in[3] = {L[i] <= sh, mid, L[i] >= hl};
        for (int z = 0; z < 3; ++z)
            if (in[z]) sum[z][0] += L[i], sum[z][1] += A[i], sum[z][2] += B[i], cnt[z] += 1.0;
    }
    if (!mids.empty() && mids.size() >= need) {
        std::sort(mids.begin(), mids.end());
        size_t m = mids.size();
        st.has_sat = true;
        st.sat = m % 2 ? mids[m / 2] : (mids[m / 2 - 1] + mids[m / 2]) / 2.0;
    }
    for (int z = 0; z < 3; ++z) {
        st.has_tint[z] = cnt[z] > 0.0 && cnt[z] >= need;
        if (st.has_tint[z])
            for (int c = 0; c < 3; ++c) st.tint[z][c] = sum[z][c] / cnt[z];
    }
    return st;
}

inline Stats stats(const double* rgb, size_t n, int stride, bool srgb) {
    Stats st;
    st.nL = 9;
    std::vector<double> L(n), A(n), B(n), C(n);
    std::vector<char> ok(n);
    for (size_t i = 0; i < n; ++i) {
        V3 enc, lin;
        bool good = true;
        for (int c = 0; c < 3; ++c) {
            enc[c] = clip(sane(rgb[i * stride + c]), 0.0, 1.0);
            good = good && enc[c] < 254.0 / 255.0 && enc[c] > 1.0 / 255.0;
            lin[c] = srgb ? srgb_decode(enc[c]) : g24_decode(enc[c]);
        }
        V3 lab = lab_709(lin);
        L[i] = lab[0], A[i] = lab[1], B[i] = lab[2], C[i] = std::hypot(lab[1], lab[2]);
        ok[i] = good;
    }
    std::vector<double> sorted = L;
    std::sort(sorted.begin(), sorted.end());
    for (int k = 0; k < 9; ++k) st.L[k] = percentile(sorted, kPcts3[k]);
    double lo = st.L[3], hi = st.L[5], need = kMinShare * n;
    std::vector<double> mids;
    double sum[3][3] = {}, cnt[3] = {};
    for (size_t i = 0; i < n; ++i) {
        if (!ok[i]) continue;
        int z = L[i] <= lo ? 0 : L[i] >= hi ? 2 : 1;
        if (z == 1) mids.push_back(C[i] / (L[i] + 16.0));
        sum[z][0] += L[i], sum[z][1] += A[i], sum[z][2] += B[i], cnt[z] += 1.0;
    }
    if (!mids.empty() && mids.size() >= need) {
        std::sort(mids.begin(), mids.end());
        size_t m = mids.size();
        st.has_sat = true;
        st.sat = m % 2 ? mids[m / 2] : (mids[m / 2 - 1] + mids[m / 2]) / 2.0;
    }
    for (int z = 0; z < 3; ++z) {
        st.has_tint[z] = cnt[z] > 0.0 && cnt[z] >= need;
        if (st.has_tint[z])
            for (int c = 0; c < 3; ++c) st.tint[z][c] = sum[z][c] / cnt[z];
    }
    return st;
}

inline int fit_curve3(const double* src, const double* dst, int m, double out[9][2]) {
    double x[9], y[9], k[9][2];
    for (int i = 0; i < m; ++i) x[i] = clip(src[i], 0.0, 100.0), y[i] = clip(dst[i], 0.0, 100.0);
    int nk = 0;
    for (int i = 0; i < m;) {
        int j = i + 1;
        while (j < m && x[j] - x[i] < kMerge) ++j;
        double sx = 0.0, sy = 0.0;
        for (int t = i; t < j; ++t) sx += x[t], sy += y[t];
        k[nk][0] = sx / (j - i), k[nk][1] = sy / (j - i), ++nk;
        i = j;
    }
    if (nk == 1) k[1][0] = k[0][0] + kMerge, k[1][1] = k[0][1] + kMerge, nk = 2;
    out[0][0] = k[0][0], out[0][1] = k[0][1];
    for (int i = 1; i < nk; ++i) {
        double dx = k[i][0] - out[i - 1][0], py = out[i - 1][1];
        out[i][0] = k[i][0];
        out[i][1] = std::min(std::max(k[i][1], py + kSlope3Min * dx), py + kSlope3Max * dx);
    }
    return nk;
}

inline double curve_eval3(const State& st, double L) {
    const double(*k)[2] = st.curve;
    int n = st.n;
    if (L < k[0][0]) return std::max(k[0][1] + (L - k[0][0]) * ((k[1][1] - k[0][1]) / (k[1][0] - k[0][0])), 0.0);
    if (L > k[n - 1][0]) return k[n - 1][1] + (L - k[n - 1][0]) * kSlope3Min;
    int j = 0;
    while (j < n - 2 && !(L < k[j + 1][0])) ++j;
    double t = (L - k[j][0]) / (k[j + 1][0] - k[j][0]);
    return k[j][1] + t * (k[j + 1][1] - k[j][1]);
}

inline V3 gain_vec(double r, double b) {
    double s = k709ToXyz[3] * r + k709ToXyz[4] + k709ToXyz[5] * b;
    return {r / s, 1.0 / s, b / s};
}

inline void solve_gains(double L, double da, double db, double* out) {
    double y = Y_of_L(L), h = 1e-4, v[2] = {0.0, 0.0};
    V3 base = lab_709({y, y, y});
    double ta = base[1] + da, tb = base[2] + db;
    auto ab = [&](double lr, double lb, double* o) {
        V3 g = gain_vec(std::exp(lr), std::exp(lb));
        V3 lab = lab_709({y * g[0], y * g[1], y * g[2]});
        o[0] = lab[1] - ta, o[1] = lab[2] - tb;
    };
    for (int it = 0; it < kNewtonSteps; ++it) {
        double f[2], fr[2], fb[2];
        ab(v[0], v[1], f);
        ab(v[0] + h, v[1], fr);
        ab(v[0], v[1] + h, fb);
        double jr[2] = {(fr[0] - f[0]) / h, (fr[1] - f[1]) / h}, jb[2] = {(fb[0] - f[0]) / h, (fb[1] - f[1]) / h};
        double det = jr[0] * jb[1] - jb[0] * jr[1];
        if (!(std::fabs(det) > 1e-12)) break;
        double s0 = (f[0] * jb[1] - jb[0] * f[1]) / det, s1 = (jr[0] * f[1] - f[0] * jr[1]) / det;
        v[0] = clip(v[0] - s0, std::log(kGainMin), std::log(kGainMax));
        v[1] = clip(v[1] - s1, std::log(kGainMin), std::log(kGainMax));
    }
    out[0] = std::exp(v[0]), out[1] = std::exp(v[1]);
}

inline void knots(const double* in, double* v) {
    for (int i = 0; i < 7; ++i) v[i] = clip(in[i], kKnotGap, 100.0 - kKnotGap * 8);
    for (int i = 1; i < 7; ++i) v[i] = std::max(v[i], v[i - 1] + kKnotGap);
}

inline void fit_curve(const double* src, const double* dst, double out[7][2]) {
    double x[7], y[7], px = 0.0, py = 0.0;
    knots(src, x);
    knots(dst, y);
    for (int i = 0; i < 7; ++i) {
        double dx = x[i] - px;
        double lo = std::max(py + kSlopeMin * dx, 100.0 - kSlopeMax * (100.0 - x[i]));
        double hi = std::min(py + kSlopeMax * dx, 100.0 - kSlopeMin * (100.0 - x[i]));
        px = x[i], py = clip(y[i], lo, hi);
        out[i][0] = px, out[i][1] = py;
    }
}

inline void tint_gains(double L, double da, double db, double* out) {
    double y = Y_of_L(L), h = 1e-3;
    V3 base = lab_709({y, y, y}), r = lab_709({y * (1.0 + h), y, y}), b = lab_709({y, y, y * (1.0 + h)});
    double jr[2] = {(r[1] - base[1]) / h, (r[2] - base[2]) / h}, jb[2] = {(b[1] - base[1]) / h, (b[2] - base[2]) / h};
    double det = jr[0] * jb[1] - jb[0] * jr[1];
    double d0 = (da * jb[1] - jb[0] * db) / det, d1 = (jr[0] * db - da * jr[1]) / det;
    out[0] = clip(1.0 + d0, kTintMin, kTintMax);
    out[1] = clip(1.0 + d1, kTintMin, kTintMax);
}

inline State fit_v2(const Stats& frame, const Stats& ref, const std::string& source) {
    State st;
    st.source = source;
    st.frame = frame;
    st.ref = ref;
    st.sat = frame.has_sat && ref.has_sat && frame.sat != 0.0 && ref.sat != 0.0 ? clip(ref.sat / frame.sat, kSatMin, kSatMax) : 1.0;
    for (int z = 0; z < 3; ++z) {
        if (frame.has_tint[z] && ref.has_tint[z])
            tint_gains(frame.tint[z][0], ref.tint[z][1] - frame.tint[z][1], ref.tint[z][2] - frame.tint[z][2], st.tint[z]);
        else
            st.tint[z][0] = st.tint[z][1] = 1.0;
    }
    fit_curve(frame.L, ref.L, st.curve);
    double sh = frame.has_tint[0] ? frame.tint[0][0] : std::min(frame.L[1], 40.0);
    double hl = frame.has_tint[2] ? frame.tint[2][0] : std::max(frame.L[5], 55.0);
    double mid = frame.has_tint[1] ? frame.tint[1][0] : frame.L[3];
    st.zone_L[0] = sh, st.zone_L[1] = std::min(std::max(mid, sh + 1.0), hl - 1.0), st.zone_L[2] = hl;
    return st;
}

inline void monotone_tint(State& st);

inline State fit(const Stats& frame, const Stats& ref, const std::string& source) {
    State st;
    st.version = 3;
    st.source = source;
    st.frame = frame;
    st.ref = ref;
    st.n = fit_curve3(frame.L, ref.L, 9, st.curve);
    st.sat = frame.has_sat && ref.has_sat && frame.sat > 1e-6 ? clip(ref.sat / frame.sat, kSat3Min, kSat3Max) : 1.0;
    for (int z = 0; z < 3; ++z) {
        st.tint[z][0] = st.tint[z][1] = 1.0;
        if (!(frame.has_tint[z] && ref.has_tint[z])) continue;
        double L2 = curve_eval3(st, frame.tint[z][0]);
        double keep = st.sat * (L2 + 16.0) / (frame.tint[z][0] + 16.0);
        solve_gains(L2, ref.tint[z][1] - keep * frame.tint[z][1], ref.tint[z][2] - keep * frame.tint[z][2], st.tint[z]);
    }
    double sh = frame.has_tint[0] ? frame.tint[0][0] : frame.L[2];
    double hl = frame.has_tint[2] ? frame.tint[2][0] : frame.L[6];
    double mid = frame.has_tint[1] ? frame.tint[1][0] : frame.L[4];
    if (hl - sh < 2.0) sh = std::min(std::max(sh - 1.0, 0.0), 98.0), hl = sh + 2.0;
    st.zone_L[0] = sh, st.zone_L[1] = std::min(std::max(mid, sh + 1.0), hl - 1.0), st.zone_L[2] = hl;
    monotone_tint(st);
    return st;
}

inline double curve_eval(const State& st, double L) {
    double kx[9], ky[9];
    kx[0] = ky[0] = 0.0;
    for (int i = 0; i < 7; ++i) kx[i + 1] = st.curve[i][0], ky[i + 1] = st.curve[i][1];
    kx[8] = ky[8] = 100.0;
    if (L > 100.0) return 100.0 + (L - 100.0) * ((ky[8] - ky[7]) / (kx[8] - kx[7]));
    if (L <= kx[0]) return ky[0];
    if (L >= kx[8]) return ky[8];
    int j = 0;
    while (j < 7 && !(L < kx[j + 1])) ++j;
    double slope = (ky[j + 1] - ky[j]) / (kx[j + 1] - kx[j]);
    return slope * (L - kx[j]) + ky[j];
}

inline V3 transform3(const State& st, const V3& c) {
    bool identity = st.n == 2 && st.curve[0][0] == 0.0 && st.curve[0][1] == 0.0 &&
                    st.curve[1][0] == 100.0 && st.curve[1][1] == 100.0 && st.sat == 1.0;
    for (int z = 0; z < 3; ++z) identity = identity && st.tint[z][0] == 1.0 && st.tint[z][1] == 1.0;
    if (identity) return c;
    V3 rgb = mul(kDwgTo709, c);
    // Match-only analysis measures the clipped Rec.709 display image.
    if (st.source == "only") for (double& v : rgb) v = clip(v, 0.0, 1.0);
    double y = luma(rgb), ye = std::max(y, kLiftEps), L0 = L_of_Y(ye);
    double y2 = Y_of_L(curve_eval3(st, L0));
    double k = std::min(y2 / ye, kLiftMax);
    for (double& v : rgb) v = y2 + k * (v - ye);
    y2 += k * (y - ye);
    for (double& v : rgb) v = y2 + st.sat * (v - y2);
    double ws = clip((st.zone_L[1] - L0) / (st.zone_L[1] - st.zone_L[0]), 0.0, 1.0);
    double wh = clip((L0 - st.zone_L[1]) / (st.zone_L[2] - st.zone_L[1]), 0.0, 1.0);
    V3 gs = gain_vec(st.tint[0][0], st.tint[0][1]), gm = gain_vec(st.tint[1][0], st.tint[1][1]), gh = gain_vec(st.tint[2][0], st.tint[2][1]);
    for (int i = 0; i < 3; ++i) rgb[i] *= (1.0 - ws - wh) * gm[i] + ws * gs[i] + wh * gh[i];
    V3 out = mul(k709ToDwg, rgb);
    for (double& v : out) v = std::max(v, 0.0);
    return out;
}

inline bool grey_monotone(const State& st) {
    V3 prev{}, prev709{};
    for (int i = 0; i < kMonoGrid; ++i) {
        double y = Y_of_L(100.0 * i / (kMonoGrid - 1));
        V3 out = transform3(st, mul(k709ToDwg, {y, y, y})), o709 = mul(kDwgTo709, out);
        if (i)
            for (int c = 0; c < 3; ++c)
                if (!(out[c] - prev[c] >= 0.0) || !(o709[c] - prev709[c] >= -1e-12)) return false;
        prev = out, prev709 = o709;
    }
    return true;
}

inline void monotone_tint(State& st) {
    double base[3][2];
    std::memcpy(base, st.tint, sizeof base);
    for (int i = 0; i <= kMonoSteps; ++i) {
        double t = 1.0 - static_cast<double>(i) / kMonoSteps;
        for (int z : {0, 2})
            for (int c = 0; c < 2; ++c) {
                double m = std::log(base[1][c]);
                st.tint[z][c] = std::exp(m + t * (std::log(base[z][c]) - m));
            }
        if (grey_monotone(st)) return;
    }
}

inline V3 transform(const State& st, const V3& c) {
    if (st.version == 3) return transform3(st, c);
    V3 rgb = mul(kDwgTo709, c);
    double y = luma(rgb), L0 = L_of_Y(y);
    if (y > 1e-6) {
        double k = Y_of_L(curve_eval(st, L_of_Y(y))) / y;
        for (double& v : rgb) v *= k;
    }
    y = luma(rgb);
    for (double& v : rgb) v = y + st.sat * (v - y);
    double ws, wh;
    if (st.version == 2) {
        ws = clip((st.zone_L[1] - L0) / (st.zone_L[1] - st.zone_L[0]), 0.0, 1.0);
        wh = clip((L0 - st.zone_L[1]) / (st.zone_L[2] - st.zone_L[1]), 0.0, 1.0);
    } else {
        double lg = std::log2(std::max(y, 1e-9) / 0.18);
        ws = clip(-lg / kShadowStops, 0.0, 1.0), wh = clip(lg / kWhiteStops, 0.0, 1.0);
    }
    V3 gm = {st.tint[1][0], 1.0, st.tint[1][1]}, gs = {st.tint[0][0], 1.0, st.tint[0][1]}, gh = {st.tint[2][0], 1.0, st.tint[2][1]};
    for (int i = 0; i < 3; ++i) rgb[i] *= (1.0 - ws - wh) * gm[i] + ws * gs[i] + wh * gh[i];
    V3 out = mul(k709ToDwg, rgb);
    for (double& v : out) v = std::max(v, 0.0);
    return out;
}

inline double smoothstep(double e0, double e1, double x) {
    double t = clip((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

inline double skin(const V3& c) {
    V3 lab = lab_709(mul(kDwgTo709, c));
    double C = std::hypot(lab[1], lab[2]);
    double h = std::fmod(std::atan2(lab[2], lab[1]) * (180.0 / 3.141592653589793), 360.0);
    if (h < 0.0) h += 360.0;
    double wh = smoothstep(10.0, 20.0, h) * (1.0 - smoothstep(75.0, 85.0, h));
    double wc = smoothstep(2.0, 8.0, C) * (1.0 - smoothstep(45.0, 51.0, C));
    return lab[0] >= 15.0 && lab[0] <= 95.0 ? wh * wc : 0.0;
}

inline V3 blended(const State& st, double strength, double protect, const V3& c) {
    double w = strength * (1.0 - protect * skin(c));
    V3 m = transform(st, c);
    return {c[0] + w * (m[0] - c[0]), c[1] + w * (m[1] - c[1]), c[2] + w * (m[2] - c[2])};
}

inline State refine(State st, const double* encoded, size_t count, int stride, const double* reference_encoded = nullptr) {
    bool close = true;
    for (int i = 0; i < 9; ++i) close = close && std::fabs(st.frame.L[i] - st.ref.L[i]) < 0.5;
    if (st.frame.has_sat && st.ref.has_sat) close = close && std::fabs(st.frame.sat - st.ref.sat) < 0.005;
    for (int z = 0; z < 3; ++z)
        if (st.frame.has_tint[z] && st.ref.has_tint[z])
            close = close && std::hypot(st.frame.tint[z][1] - st.ref.tint[z][1], st.frame.tint[z][2] - st.ref.tint[z][2]) < 0.5;
    // Quantization can move flat patches across percentile zone boundaries.
    // The caller supplies spatially aligned sRGB samples only when dimensions match.
    if (reference_encoded && count) {
        double total = 0.0;
        size_t near_count = 0;
        for (size_t i = 0; i < count; ++i) {
            V3 a{}, b{};
            for (int c = 0; c < 3; ++c) {
                a[c] = g24_decode(clip(sane(encoded[i * stride + c]), 0.0, 1.0));
                b[c] = srgb_decode(clip(sane(reference_encoded[i * 3 + c]), 0.0, 1.0));
            }
            a = lab_709(a), b = lab_709(b);
            double d = std::sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]));
            total += d;
            near_count += d < 1.0;
        }
        close = close || (total / count < 0.5 && static_cast<double>(near_count) / count >= 0.95);
    }
    if (close) {
        st.n = 2;
        st.curve[0][0] = st.curve[0][1] = 0.0;
        st.curve[1][0] = st.curve[1][1] = 100.0;
        st.sat = 1.0;
        for (auto& tint : st.tint) tint[0] = tint[1] = 1.0;
        return st;
    }
    std::vector<V3> samples;
    for (size_t i = 0; i < count; i += 16) {
        const double* p = encoded + i * stride;
        samples.push_back(mul(k709ToDwg, {g24_decode(sane(p[0])), g24_decode(sane(p[1])), g24_decode(sane(p[2]))}));
    }
    std::vector<double> output(samples.size() * 3);
    auto cost = [&](const State& candidate) {
        for (size_t i = 0; i < samples.size(); ++i) {
            V3 out = mul(kDwgTo709, transform(candidate, samples[i]));
            for (int c = 0; c < 3; ++c) output[i * 3 + c] = g24_encode(out[c]);
        }
        Stats measured = stats(output.data(), samples.size(), 3, false);
        double error = 0.0, tint_error = 0.0;
        for (int i = 0; i < 9; ++i) error += std::fabs(measured.L[i] - st.ref.L[i]) / 9.0;
        if (measured.has_sat && st.ref.has_sat) error += 66.0 * std::fabs(measured.sat - st.ref.sat);
        int zones = 0;
        for (int z = 0; z < 3; ++z) {
            if (!st.ref.has_tint[z]) continue;
            ++zones;
            tint_error += measured.has_tint[z] ? std::hypot(measured.tint[z][1] - st.ref.tint[z][1],
                                                          measured.tint[z][2] - st.ref.tint[z][2]) : 100.0;
        }
        return error + (zones ? tint_error / zones : 0.0);
    };
    // Cast and chroma are coupled. Try the original and a low-saturation seed.
    State winner;
    double winner_error = 1e300;
    for (double scale : {1.0, 0.15}) {
        State candidate = st;
        candidate.sat = clip(st.sat * scale, kSat3Min, kSat3Max);
        for (int z = 0; z < 3; ++z) {
            if (!st.frame.has_tint[z] || !st.ref.has_tint[z]) continue;
            double L2 = curve_eval3(candidate, st.frame.tint[z][0]);
            double keep = candidate.sat * (L2 + 16.0) / (st.frame.tint[z][0] + 16.0);
            solve_gains(L2, st.ref.tint[z][1] - keep * st.frame.tint[z][1],
                                   st.ref.tint[z][2] - keep * st.frame.tint[z][2], candidate.tint[z]);
        }
        monotone_tint(candidate);
        double best = cost(candidate);
        for (double step : {0.5, 0.25, 0.12, 0.06, 0.03})
            for (int repeat = 0; repeat < 3; ++repeat)
                // Saturation, six zone gains, then two whole-image gain moves.
                for (int index = 0; index < 9; ++index) {
                    State base = candidate;
                    for (int sign : {-1, 1}) {
                        State trial = base;
                        if (index == 0) trial.sat = clip(base.sat * std::exp(sign * step), kSat3Min, kSat3Max);
                        else {
                            int c = (index - 1) % 2;
                            int begin = index > 6 ? 0 : (index - 1) / 2;
                            int end = index > 6 ? 3 : begin + 1;
                            for (int z = begin; z < end; ++z)
                                trial.tint[z][c] = clip(base.tint[z][c] * std::exp(sign * step), kGainMin, kGainMax);
                        }
                        // Do not undo the monotonicity established by fit().
                        if (!grey_monotone(trial)) continue;
                        double error = cost(trial);
                        if (error < best - 0.01) candidate = trial, best = error;
                    }
                }
        if (best < winner_error) winner = candidate, winner_error = best;
    }
    return winner;
}

inline double grid(int i, int n) { return i == n - 1 ? 1.0 : i * (1.0 / (n - 1)); }

inline std::vector<float> bake_lut(const State& st, double strength, double protect, int n = kLutSize) {
    std::vector<float> out(static_cast<size_t>(n) * n * n * 3);
    for (int b = 0; b < n; ++b)
        for (int g = 0; g < n; ++g)
            for (int r = 0; r < n; ++r) {
                V3 v = blended(st, strength, protect, {di_decode(grid(r, n)), di_decode(grid(g, n)), di_decode(grid(b, n))});
                float* o = &out[((static_cast<size_t>(b) * n + g) * n + r) * 3];
                for (int c = 0; c < 3; ++c) o[c] = static_cast<float>(di_encode(v[c]));
            }
    return out;
}

inline std::string cube_text(const State& st, double strength, double protect, int n = kLutSize) {
    std::string s = "TITLE \"FIXER reference match\"\n# " + std::string(kCubeHeader) + "\nLUT_3D_SIZE " + std::to_string(n) +
                    "\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n";
    char line[96];
    for (int b = 0; b < n; ++b)
        for (int g = 0; g < n; ++g)
            for (int r = 0; r < n; ++r) {
                V3 lin = mul(k709ToDwg, {g24_decode(grid(r, n)), g24_decode(grid(g, n)), g24_decode(grid(b, n))});
                V3 o = mul(kDwgTo709, blended(st, strength, protect, lin));
                char* q = line;
                for (int c = 0; c < 3; ++c) {
                    q = std::to_chars(q, line + sizeof line - 2, g24_encode(o[c]), std::chars_format::fixed, 6).ptr;
                    *q++ = c < 2 ? ' ' : '\n';
                }
                s.append(line, q);
            }
    return s;
}

inline void put_num(std::string& s, double v) {
    char buf[40];
    s.append(buf, std::to_chars(buf, buf + sizeof buf, v).ptr);
}

inline void put_list(std::string& s, const double* v, int n) {
    s += '[';
    for (int i = 0; i < n; ++i) {
        if (i) s += ',';
        put_num(s, v[i]);
    }
    s += ']';
}

inline void put_stats(std::string& s, const Stats& st) {
    s += "{\"L\":";
    put_list(s, st.L, st.nL);
    s += ",\"sat\":";
    if (st.has_sat) put_num(s, st.sat);
    else s += "null";
    s += ",\"tint\":{";
    for (int z = 0; z < 3; ++z) {
        s += std::string(z ? "," : "") + "\"" + kZones[z] + "\":";
        if (st.has_tint[z]) put_list(s, st.tint[z], 3);
        else s += "null";
    }
    s += "}}";
}

inline std::string to_json(const State& st) {
    std::string s = "{\"version\":" + std::to_string(st.version) + ",\"mode\":\"simple\",\"source\":\"" + st.source + "\",\"curve\":[";
    for (int i = 0; i < st.n; ++i) {
        if (i) s += ',';
        put_list(s, st.curve[i], 2);
    }
    s += "],\"sat\":";
    put_num(s, st.sat);
    s += ",\"tint\":{";
    for (int z = 0; z < 3; ++z) {
        s += std::string(z ? "," : "") + "\"" + kZones[z] + "\":";
        put_list(s, st.tint[z], 2);
    }
    s += "}";
    if (st.version >= 2) {
        s += ",\"zone_L\":";
        put_list(s, st.zone_L, 3);
    }
    s += ",\"ref\":";
    put_stats(s, st.ref);
    s += ",\"frame\":";
    put_stats(s, st.frame);
    s += "}";
    return s;
}

struct Json {
    enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
    double num = 0.0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;
    const Json* get(const char* key) const {
        for (const auto& kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

struct JsonReader {
    const char* p;
    const char* e;
    void ws() {
        while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }
    bool lit(const char* s) {
        size_t n = std::strlen(s);
        if (static_cast<size_t>(e - p) < n || std::strncmp(p, s, n)) return false;
        p += n;
        return true;
    }
    bool string(std::string& out) {
        if (p >= e || *p != '"') return false;
        for (++p; p < e && *p != '"'; ++p) {
            if (*p != '\\') {
                out += *p;
                continue;
            }
            if (++p >= e) return false;
            char c = *p;
            if (c == 'u') {
                if (e - p < 5) return false;
                p += 4;
                out += '?';
            } else {
                out += c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c == 'b' ? '\b' : c == 'f' ? '\f' : c;
            }
        }
        if (p >= e) return false;
        ++p;
        return true;
    }
    bool value(Json& v, int depth) {
        if (depth > 32) return false;
        ws();
        if (p >= e) return false;
        if (*p == '{') {
            v.kind = Json::Obj;
            ++p;
            ws();
            if (p < e && *p == '}') return ++p, true;
            while (true) {
                ws();
                std::pair<std::string, Json> kv;
                if (!string(kv.first)) return false;
                ws();
                if (p >= e || *p != ':') return false;
                ++p;
                if (!value(kv.second, depth + 1)) return false;
                v.obj.push_back(std::move(kv));
                ws();
                if (p < e && *p == ',') {
                    ++p;
                    continue;
                }
                if (p < e && *p == '}') return ++p, true;
                return false;
            }
        }
        if (*p == '[') {
            v.kind = Json::Arr;
            ++p;
            ws();
            if (p < e && *p == ']') return ++p, true;
            while (true) {
                v.arr.emplace_back();
                if (!value(v.arr.back(), depth + 1)) return false;
                ws();
                if (p < e && *p == ',') {
                    ++p;
                    continue;
                }
                if (p < e && *p == ']') return ++p, true;
                return false;
            }
        }
        if (*p == '"') return v.kind = Json::Str, string(v.str);
        if (lit("null")) return v.kind = Json::Null, true;
        if (lit("true")) return v.kind = Json::Bool, v.num = 1.0, true;
        if (lit("false")) return v.kind = Json::Bool, v.num = 0.0, true;
        if (*p == '+') return false;
        auto r = std::from_chars(p, e, v.num);
        if (r.ec != std::errc() || r.ptr == p) return false;
        v.kind = Json::Num;
        p = r.ptr;
        return true;
    }
};

inline bool parse_json(const std::string& text, Json& out) {
    JsonReader r{text.c_str(), text.c_str() + text.size()};
    if (!r.value(out, 0)) return false;
    r.ws();
    return r.p == r.e;
}

inline bool numbers(const Json* j, double* out, int n) {
    if (!j || j->kind != Json::Arr || static_cast<int>(j->arr.size()) != n) return false;
    for (int i = 0; i < n; ++i) {
        if (j->arr[i].kind != Json::Num || !std::isfinite(j->arr[i].num)) return false;
        out[i] = j->arr[i].num;
    }
    return true;
}

inline bool fail_why(std::string* why, const char* text) {
    if (why) *why = text;
    return false;
}

inline bool in_range(double v, double lo, double hi) { return v >= lo && v <= hi; }

inline bool parse_v3(const Json& j, State& st, std::string* why) {
    const Json *src = j.get("source"), *curve = j.get("curve"), *sat = j.get("sat"), *tint = j.get("tint"), *zl = j.get("zone_L");
    if (!curve || curve->kind != Json::Arr || curve->arr.size() < 2 || curve->arr.size() > 9) return fail_why(why, "the curve needs 2 to 9 points");
    State out;
    out.version = 3;
    out.n = static_cast<int>(curve->arr.size());
    for (int i = 0; i < out.n; ++i)
        if (!numbers(&curve->arr[i], out.curve[i], 2)) return fail_why(why, "a curve point is not two numbers");
    for (int i = 0; i < out.n; ++i) {
        double x = out.curve[i][0], y = out.curve[i][1];
        if (!in_range(x, 0.0, 100.5)) return fail_why(why, "a curve x is outside [0, 100.5]");
        if (!in_range(y, 0.0, 125.0)) return fail_why(why, "a curve y is outside [0, 125]");
        if (i == 0) continue;
        double dx = x - out.curve[i - 1][0];
        if (!(dx > 0.0)) return fail_why(why, "the curve x values are not strictly increasing");
        if (!in_range((y - out.curve[i - 1][1]) / dx, kSlope3Min - 1e-9, kSlope3Max + 1e-9)) return fail_why(why, "a curve slope is outside [0.2, 5]");
    }
    if (!sat || sat->kind != Json::Num || !in_range(sat->num, kSat3Min, kSat3Max)) return fail_why(why, "saturation is outside [0.05, 2.5]");
    if (!tint || tint->kind != Json::Obj) return fail_why(why, "the tints are missing");
    for (int z = 0; z < 3; ++z)
        if (!numbers(tint->get(kZones[z]), out.tint[z], 2) || !in_range(out.tint[z][0], kGainMin, kGainMax) ||
            !in_range(out.tint[z][1], kGainMin, kGainMax))
            return fail_why(why, "a tint is outside [0.015625, 64]");
    if (!numbers(zl, out.zone_L, 3) ||
        !(out.zone_L[0] >= 0.0 && out.zone_L[0] < out.zone_L[1] && out.zone_L[1] < out.zone_L[2] && out.zone_L[2] <= 100.0))
        return fail_why(why, "zone_L needs three increasing L* values in [0, 100]");
    out.source = src->str;
    out.sat = sat->num;
    st = out;
    return true;
}

inline bool parse_state(const std::string& text, State& st, std::string* why = nullptr) {
    Json j;
    if (!parse_json(text, j) || j.kind != Json::Obj) return fail_why(why, "not valid JSON");
    const Json *ver = j.get("version"), *mode = j.get("mode"), *src = j.get("source"), *curve = j.get("curve"),
               *sat = j.get("sat"), *tint = j.get("tint"), *zl = j.get("zone_L");
    if (!ver || ver->kind != Json::Num || (ver->num != 1.0 && ver->num != 2.0 && ver->num != 3.0)) return fail_why(why, "unknown version");
    if (!mode || mode->kind != Json::Str || mode->str != "simple") return fail_why(why, "unknown mode");
    if (!src || src->kind != Json::Str || (src->str != "film" && src->str != "only")) return fail_why(why, "unknown order");
    if (ver->num == 3.0) return parse_v3(j, st, why);
    if (!curve || curve->kind != Json::Arr || curve->arr.size() != 7) return fail_why(why, "the curve needs 7 points");
    State out;
    out.version = static_cast<int>(ver->num);
    for (int i = 0; i < 7; ++i)
        if (!numbers(&curve->arr[i], out.curve[i], 2)) return fail_why(why, "a curve point is not two numbers");
    double px = 0.0, py = 0.0;
    for (int i = 0; i <= 7; ++i) {
        double x = i < 7 ? out.curve[i][0] : 100.0, y = i < 7 ? out.curve[i][1] : 100.0;
        if (i < 7 && !(x > 0.0 && x < 100.0)) return fail_why(why, "a curve x is outside (0, 100)");
        if (i < 7 && !in_range(y, 0.0, 100.0)) return fail_why(why, "a curve y is outside [0, 100]");
        if (!(x > px)) return fail_why(why, "the curve x values are not strictly increasing");
        if (!in_range((y - py) / (x - px), kSlopeMin - 1e-9, kSlopeMax + 1e-9)) return fail_why(why, "a curve slope is outside [0.5, 2]");
        px = x, py = y;
    }
    if (!sat || sat->kind != Json::Num || !in_range(sat->num, kSatMin, kSatMax)) return fail_why(why, "saturation is outside [0.85, 1.25]");
    if (!tint || tint->kind != Json::Obj) return fail_why(why, "the tints are missing");
    for (int z = 0; z < 3; ++z)
        if (!numbers(tint->get(kZones[z]), out.tint[z], 2) || !in_range(out.tint[z][0], kTintMin, kTintMax) ||
            !in_range(out.tint[z][1], kTintMin, kTintMax))
            return fail_why(why, "a tint is outside [0.93, 1.07]");
    if (out.version == 2 && (!numbers(zl, out.zone_L, 3) || !(out.zone_L[0] >= 0.0 && out.zone_L[0] < out.zone_L[1] &&
                                                             out.zone_L[1] < out.zone_L[2] && out.zone_L[2] <= 100.0)))
        return fail_why(why, "zone_L needs three increasing L* values in [0, 100]");
    out.source = src->str;
    out.sat = sat->num;
    st = out;
    return true;
}

}
