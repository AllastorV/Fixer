#pragma once
#include <cstddef>
#include <cstdint>

namespace fe {

struct RenderParams {
    int32_t in_space;
    int32_t out_space;
    float exposure_ev;
    float push_pull;
    float impact;
    int32_t premultiplied;
    int32_t lut_size;
    int32_t reserved;
    float grain_amount;
    float grain_sigma_px;
    uint32_t grain_seed;
    int32_t frame;
    int32_t origin_x;
    int32_t origin_y;
    float halation_amount;
    float halation_radius_px;
    float bloom_amount;
    float bloom_radius_px;
    float print_log_offset[3];
    int32_t spatial_half;
    int32_t reserved2[1];
    float match_protect;
};

static_assert(sizeof(RenderParams) == 96, "RenderParams ABI");
static_assert(offsetof(RenderParams, print_log_offset) == 72, "RenderParams ABI");
static_assert(offsetof(RenderParams, halation_amount) == 56, "RenderParams ABI");
static_assert(offsetof(RenderParams, exposure_ev) == 8, "RenderParams ABI");
static_assert(offsetof(RenderParams, lut_size) == 24, "RenderParams ABI");
static_assert(offsetof(RenderParams, grain_amount) == 32, "RenderParams ABI");
static_assert(offsetof(RenderParams, origin_y) == 52, "RenderParams ABI");

constexpr float kPrinterPoint = 0.025f;
constexpr float kGrainCloudRadius = 2.5f;
constexpr float kGrainCloudMean = 1.0471976f;
constexpr float kGrainCloudStd = 0.91529123f;
constexpr float kGrainBaseSigmaLog10 = 0.04f;
constexpr float kGrainTonalFloor = 0.25f;
constexpr float kGrainTonalWidth = 1.0f;
constexpr float kGrainMinBlurPx = 0.5f;
constexpr float kGrainMaxBlurPx = 20.0f;
constexpr float kGrainRefHeight = 1080.0f;
constexpr int kGrainTableSize = 64;
constexpr float kGrainTableLo = -3.0f;
constexpr float kGrainTableHi = 3.0f;
constexpr float kHalationWeight[3] = {1.0f, 0.25f, 0.0f};
constexpr float kHalationNarrow = 0.3f;
constexpr float kHalationMix = 0.5f;
constexpr float kBloomRadiusRef = 15.0f;
constexpr float kSpatialMaxSigmaPx = 120.0f;

}
