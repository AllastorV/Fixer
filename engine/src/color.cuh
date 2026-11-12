#pragma once
#include <cuda_runtime.h>

#include <cmath>

#include "spaces_gen.h"

namespace fe {

#define FE_HD __host__ __device__ __forceinline__

constexpr float kDiA = 0.0075f, kDiB = 7.0f, kDiC = 0.07329248f, kDiM = 10.44426855f;
constexpr float kDiLinCut = 0.00262409f, kDiLogCut = 0.02740668f;
constexpr float kCctLinCut = 0.0078125f, kCctLogCut = 0.155251141552511f;
constexpr float kCctA = 10.5402377416545f, kCctB = 0.0729055341958355f;
constexpr float kFMax = 65504.0f;

FE_HD float sanitize(float x) {
    if (isnan(x)) return 0.0f;
    return fminf(fmaxf(x, -kFMax), kFMax);
}

FE_HD float di_encode(float x) { return x <= kDiLinCut ? kDiM * x : kDiC * (log2f(x + kDiA) + kDiB); }
FE_HD float di_decode(float y) { return y <= kDiLogCut ? y / kDiM : exp2f(y / kDiC - kDiB) - kDiA; }
FE_HD float cct_encode(float x) { return x <= kCctLinCut ? kCctA * x + kCctB : (log2f(x) + 9.72f) / 17.52f; }
FE_HD float cct_decode(float y) { return y <= kCctLogCut ? (y - kCctB) / kCctA : exp2f(y * 17.52f - 9.72f); }
FE_HD float g24_encode(float x) { return copysignf(powf(fabsf(x), 1.0f / 2.4f), x); }
FE_HD float g24_decode(float y) { return copysignf(powf(fabsf(y), 2.4f), y); }

FE_HD float encode(int space, float x) {
    switch (space) {
        case kSpace_dwg_di: return di_encode(x);
        case kSpace_acescct: return cct_encode(x);
        case kSpace_rec709_g24: return g24_encode(x);
        default: return x;
    }
}

FE_HD float decode(int space, float y) {
    switch (space) {
        case kSpace_dwg_di: return di_decode(y);
        case kSpace_acescct: return cct_decode(y);
        case kSpace_rec709_g24: return g24_decode(y);
        default: return y;
    }
}

FE_HD float3 mul(const float* m, float3 v) {
    return make_float3(m[0] * v.x + m[1] * v.y + m[2] * v.z, m[3] * v.x + m[4] * v.y + m[5] * v.z,
                       m[6] * v.x + m[7] * v.y + m[8] * v.z);
}

FE_HD float smoothstep01(float t) { return t * t * (3.0f - 2.0f * t); }

}
