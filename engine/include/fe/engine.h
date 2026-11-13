#pragma once
#include <cuda_runtime.h>

#include <string>

#include "fe/params.h"

namespace fe {

struct Rect {
    int x1, y1, x2, y2;
    int width() const { return x2 - x1; }
    int height() const { return y2 - y1; }
};

struct Image {
    const char* base;
    size_t pitch;
    Rect bounds;
};

class LutSet {
public:
    LutSet() = default;
    ~LutSet();
    LutSet(const LutSet&) = delete;
    LutSet& operator=(const LutSet&) = delete;

    static constexpr int kPrintSlot = 3;
    static constexpr int kMatchSlot = 4;
    static constexpr int kMatchOn = 1;
    static constexpr int kFilmBypass = 2;
    cudaError_t upload(int slot, const float* rgb, int size);
    void borrow_match(const LutSet& from) {
        if (!match_borrowed_ && arr_[kMatchSlot]) return;
        tex_[kMatchSlot] = from.tex_[kMatchSlot];
        match_size_ = from.match_size_;
        match_borrowed_ = true;
    }
    void set_print_domain(float lo, float hi) { print_lo_ = lo, print_hi_ = hi; }
    bool ready() const;
    bool two_stage() const { return tex_[kPrintSlot] != 0 && print_hi_ > print_lo_; }
    cudaTextureObject_t texture(int slot) const { return tex_[slot]; }
    int size() const { return size_; }
    int match_size() const { return match_size_; }
    float print_lo() const { return print_lo_; }
    float print_hi() const { return print_hi_; }
    void set_grain_table(const float* rgb) {
        for (int i = 0; i < 3 * kGrainTableSize; ++i) grain_[i] = rgb[i];
        has_grain_ = true;
    }
    const float* grain_table() const { return has_grain_ ? grain_ : nullptr; }
    void set_grain_mono(bool mono) { grain_mono_ = mono; }
    bool grain_mono() const { return grain_mono_; }

private:
    float grain_[3 * kGrainTableSize] = {};
    bool has_grain_ = false;
    bool grain_mono_ = false;
    cudaArray_t arr_[5] = {};
    cudaTextureObject_t tex_[5] = {};
    int size_ = 0;
    int match_size_ = 0;
    bool match_borrowed_ = false;
    float print_lo_ = 0.0f, print_hi_ = 0.0f;
};

cudaError_t render(const float* src, size_t src_pitch, float* dst, size_t dst_pitch, int width, int height,
                   const RenderParams& params, const LutSet& luts, cudaStream_t stream);

cudaError_t render_region(Image src, float* dst_base, size_t dst_pitch, Rect dst_bounds, Rect window,
                          const RenderParams& params, const LutSet& luts, cudaStream_t stream);

}
