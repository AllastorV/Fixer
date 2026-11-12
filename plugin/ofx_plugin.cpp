#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <map>
#include <sstream>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "fe/engine.h"
#include "ofxCore.h"
#include "ofxGPURender.h"
#include "ofxImageEffect.h"
#include "ofxMessage.h"
#include "ofxParam.h"
#include "ofxProperty.h"

namespace {

constexpr const char* kPluginId = "com.alpcavas.fixer";
constexpr const char* kDefaultProfile = "kodak-vision3-250d-2383";
constexpr const char* kSpaceLabels[] = {"DaVinci Wide Gamut / Intermediate", "DaVinci Wide Gamut / Linear", "ACEScct",
                                        "Rec.709 / Gamma 2.4"};

OfxHost* gHost = nullptr;
OfxPropertySuiteV1* gProp = nullptr;
OfxImageEffectSuiteV1* gEffect = nullptr;
OfxParameterSuiteV1* gParam = nullptr;
OfxMessageSuiteV1* gMessage = nullptr;

struct Profile {
    std::string id;
    std::string label;
    int size = 0;
    std::vector<float> slots[4];
    bool two_stage = false;
    float print_lo = 0.0f, print_hi = 0.0f;
    std::vector<float> grain;
    std::vector<float> alt[3];
    std::string alt_light;
    std::vector<float> bal[4];
    std::vector<float> scan[2];
    bool test = false;
    bool grain_mono = false;
};

bool read_lut(const std::wstring& path, size_t floats, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || static_cast<size_t>(f.tellg()) != floats * sizeof(float)) return false;
    f.seekg(0);
    out.resize(floats);
    if (!f.read(reinterpret_cast<char*>(out.data()), floats * sizeof(float))) return false;
    for (float v : out)
        if (!std::isfinite(v)) return false;
    return true;
}

const char* const kGrainRateLabels[4] = {"Every frame (cinema)", "Every 2 frames", "Every 4 frames", "Static"};
const int kGrainRateSteps[4] = {1, 2, 4, 0};
constexpr const char* kLightLabels[] = {"Matched to film", "Daylight 5500K", "Tungsten 3200K"};
constexpr const char* kGlowQualityLabels[] = {"Full", "Fast (half resolution)"};
constexpr const char* kLightIds[] = {"", "daylight", "tungsten"};

std::vector<Profile> gProfiles;

struct Instance {
    std::mutex lock;
    std::map<int, std::unique_ptr<fe::LutSet>> luts;
    int device = -1;
};

std::wstring resources_dir() {
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&resources_dir), &self);
    wchar_t path[MAX_PATH * 4];
    DWORD n = GetModuleFileNameW(self, path, static_cast<DWORD>(std::size(path)));
    std::wstring p(path, n);
    p = p.substr(0, p.find_last_of(L"\\/"));
    p = p.substr(0, p.find_last_of(L"\\/"));
    return p + L"\\Resources\\";
}

std::string json_raw(const std::string& text, const std::string& key) {
    size_t k = text.find("\"" + key + "\"");
    if (k == std::string::npos) return {};
    size_t c = text.find(':', k);
    if (c == std::string::npos) return {};
    size_t v = text.find_first_not_of(" \t\r\n", c + 1);
    if (v == std::string::npos) return {};
    if (text[v] == '"') {
        size_t e = text.find('"', v + 1);
        return e == std::string::npos ? std::string() : text.substr(v + 1, e - v - 1);
    }
    size_t e = text.find_first_of(",}\r\n", v);
    return text.substr(v, e - v);
}

bool load_profile(const std::wstring& dir, Profile& out) {
    std::ifstream meta(dir + L"\\profile.json", std::ios::binary);
    if (!meta) return false;
    std::stringstream ss;
    ss << meta.rdbuf();
    std::string text = ss.str();
    std::string kind = json_raw(text, "kind");
    out.two_stage = kind == "two_stage";
    if (kind.empty()) kind = "single";
    if (kind != "single" && kind != "two_stage") return false;
    std::string enc = json_raw(text, "lut_encoding");
    if (json_raw(text, "schema_version") != "1" || json_raw(text, "axis_order") != "r_fastest" ||
        enc != (out.two_stage ? "print_log_exposure" : "dwg_di"))
        return false;
    if (out.two_stage) {
        if (json_raw(text, "print_encoding") != "dwg_di") return false;
        out.print_lo = std::strtof(json_raw(text, "print_lo").c_str(), nullptr);
        out.print_hi = std::strtof(json_raw(text, "print_hi").c_str(), nullptr);
        if (!std::isfinite(out.print_lo) || !std::isfinite(out.print_hi) || out.print_hi <= out.print_lo) return false;
    }
    out.id = json_raw(text, "profile_id");
    out.label = json_raw(text, "label");
    out.size = std::atoi(json_raw(text, "lut_size").c_str());
    if (out.id.empty() || out.label.empty() || out.size < 2 || out.size > 129) return false;
    const wchar_t* names[4] = {L"\\under.bin", L"\\normal.bin", L"\\over.bin", L"\\print.bin"};
    size_t floats = static_cast<size_t>(out.size) * out.size * out.size * 3;
    for (int s = 0; s < (out.two_stage ? 4 : 3); ++s) {
        std::ifstream f(dir + names[s], std::ios::binary | std::ios::ate);
        if (!f || static_cast<size_t>(f.tellg()) != floats * sizeof(float)) return false;
        f.seekg(0);
        out.slots[s].resize(floats);
        if (!f.read(reinterpret_cast<char*>(out.slots[s].data()), floats * sizeof(float))) return false;
        for (float v : out.slots[s])
            if (!std::isfinite(v)) return false;
    }
    if (!json_raw(text, "balanced_slots").empty()) {
        const wchar_t* bal_names[3] = {L"\\under_bal.bin", L"\\normal_bal.bin", L"\\over_bal.bin"};
        for (int s = 0; s < 3; ++s)
            if (!read_lut(dir + bal_names[s], floats, out.bal[s])) return false;
    }
    if (json_raw(text, "balanced_print") == "print_bal.bin" && !read_lut(dir + L"\\print_bal.bin", floats, out.bal[3]))
        return false;
    if (json_raw(text, "scan_print") == "print_scan.bin" &&
        (!read_lut(dir + L"\\print_scan.bin", floats, out.scan[0]) || !read_lut(dir + L"\\print_bal_scan.bin", floats, out.scan[1])))
        return false;
    out.grain_mono = json_raw(text, "grain_mono") == "true";
    out.test = json_raw(text, "test") == "true";
    out.alt_light = json_raw(text, "alt_light");
    if (!out.alt_light.empty()) {
        if (!out.two_stage || (out.alt_light != "daylight" && out.alt_light != "tungsten")) return false;
        const wchar_t* alt_names[3] = {L"\\under_alt.bin", L"\\normal_alt.bin", L"\\over_alt.bin"};
        for (int s = 0; s < 3; ++s) {
            std::ifstream f(dir + alt_names[s], std::ios::binary | std::ios::ate);
            if (!f || static_cast<size_t>(f.tellg()) != floats * sizeof(float)) return false;
            f.seekg(0);
            out.alt[s].resize(floats);
            if (!f.read(reinterpret_cast<char*>(out.alt[s].data()), floats * sizeof(float))) return false;
            for (float v : out.alt[s])
                if (!std::isfinite(v)) return false;
        }
    }
    if (json_raw(text, "grain") == "grain.bin") {
        size_t n = 3 * static_cast<size_t>(fe::kGrainTableSize);
        std::ifstream f(dir + L"\\grain.bin", std::ios::binary | std::ios::ate);
        if (!f || static_cast<size_t>(f.tellg()) != n * sizeof(float)) return false;
        f.seekg(0);
        out.grain.resize(n);
        if (!f.read(reinterpret_cast<char*>(out.grain.data()), n * sizeof(float))) return false;
        for (float v : out.grain)
            if (!std::isfinite(v) || v < 0.0f) return false;
    }
    return true;
}

bool natural_less(const std::string& a, const std::string& b) {
    size_t i = 0, j = 0;
    auto digit = [](char c) { return c >= '0' && c <= '9'; };
    while (i < a.size() && j < b.size()) {
        if (digit(a[i]) && digit(b[j])) {
            size_t i2 = i, j2 = j;
            while (i2 < a.size() && digit(a[i2])) ++i2;
            while (j2 < b.size() && digit(b[j2])) ++j2;
            unsigned long long x = std::stoull(a.substr(i, i2 - i)), y = std::stoull(b.substr(j, j2 - j));
            if (x != y) return x < y;
            i = i2;
            j = j2;
        } else {
            if (a[i] != b[j]) return a[i] < b[j];
            ++i;
            ++j;
        }
    }
    return a.size() - i < b.size() - j;
}

void load_profiles() {
    gProfiles.clear();
    std::wstring root = resources_dir() + L"profiles\\";
    bool show_test = GetEnvironmentVariableW(L"FE_TEST_PROFILES", nullptr, 0) > 0;
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW((root + L"*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || fd.cFileName[0] == L'.') continue;
        Profile p;
        if (load_profile(root + fd.cFileName, p) && (show_test || !p.test)) gProfiles.push_back(std::move(p));
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    std::sort(gProfiles.begin(), gProfiles.end(), [](const Profile& a, const Profile& b) { return natural_less(a.label, b.label); });
}

void post_error(OfxImageEffectHandle effect, const char* text) {
    if (gMessage) gMessage->message(effect, kOfxMessageError, nullptr, "%s", text);
}

Instance* instance_of(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props;
    gEffect->getPropertySet(effect, &props);
    void* data = nullptr;
    gProp->propGetPointer(props, kOfxPropInstanceData, 0, &data);
    return static_cast<Instance*>(data);
}

void place(OfxPropertySetHandle p, const char* label, const char* hint, const char* parent) {
    gProp->propSetString(p, kOfxPropLabel, 0, label);
    if (hint) gProp->propSetString(p, kOfxParamPropHint, 0, hint);
    if (parent) gProp->propSetString(p, kOfxParamPropParent, 0, parent);
}

void define_group(OfxParamSetHandle params, const char* name, const char* label, bool open) {
    OfxPropertySetHandle p;
    gParam->paramDefine(params, kOfxParamTypeGroup, name, &p);
    gProp->propSetString(p, kOfxPropLabel, 0, label);
    gProp->propSetInt(p, kOfxParamPropGroupOpen, 0, open ? 1 : 0);
}

void define_double(OfxParamSetHandle params, const char* name, const char* label, double def, double lo, double hi,
                   const char* hint, const char* parent = nullptr) {
    OfxPropertySetHandle p;
    gParam->paramDefine(params, kOfxParamTypeDouble, name, &p);
    place(p, label, hint, parent);
    gProp->propSetDouble(p, kOfxParamPropDefault, 0, def);
    gProp->propSetDouble(p, kOfxParamPropDisplayMin, 0, lo);
    gProp->propSetDouble(p, kOfxParamPropDisplayMax, 0, hi);
    gProp->propSetDouble(p, kOfxParamPropMin, 0, lo);
    gProp->propSetDouble(p, kOfxParamPropMax, 0, hi);
}

void define_bool(OfxParamSetHandle params, const char* name, const char* label, int def, const char* hint,
                 const char* parent = nullptr) {
    OfxPropertySetHandle p;
    gParam->paramDefine(params, kOfxParamTypeBoolean, name, &p);
    place(p, label, hint, parent);
    gProp->propSetInt(p, kOfxParamPropDefault, 0, def);
}

void define_choice(OfxParamSetHandle params, const char* name, const char* label, const char* const* options, int count,
                   const char* hint, const char* parent = nullptr) {
    OfxPropertySetHandle p;
    gParam->paramDefine(params, kOfxParamTypeChoice, name, &p);
    place(p, label, hint, parent);
    for (int i = 0; i < count; ++i) gProp->propSetString(p, kOfxParamPropChoiceOption, i, options[i]);
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 0);
}

OfxStatus describe(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props;
    gEffect->getPropertySet(effect, &props);
    gProp->propSetString(props, kOfxPropLabel, 0, "FIXER");
    gProp->propSetString(props, kOfxImageEffectPluginPropGrouping, 0, "FIXER");
    gProp->propSetString(props, kOfxPropIcon, 1, "com.alpcavas.fixer.png");
    gProp->propSetString(props, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
    gProp->propSetString(props, kOfxImageEffectPropSupportedContexts, 1, kOfxImageEffectContextGeneral);
    gProp->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthFloat);
    gProp->propSetInt(props, kOfxImageEffectPropSupportsTiles, 0, 0);
    gProp->propSetInt(props, kOfxImageEffectPropSupportsMultiResolution, 0, 1);
    gProp->propSetInt(props, kOfxImageEffectPluginPropSingleInstance, 0, 0);
    gProp->propSetString(props, kOfxImageEffectPluginRenderThreadSafety, 0, kOfxImageEffectRenderFullySafe);
    gProp->propSetString(props, kOfxImageEffectPropCudaRenderSupported, 0, "true");
    gProp->propSetString(props, kOfxImageEffectPropCudaStreamSupported, 0, "true");
    return kOfxStatOK;
}

OfxStatus describe_in_context(OfxImageEffectHandle effect) {
    OfxPropertySetHandle clip;
    gEffect->clipDefine(effect, kOfxImageEffectSimpleSourceClipName, &clip);
    gProp->propSetString(clip, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
    gEffect->clipDefine(effect, kOfxImageEffectOutputClipName, &clip);
    gProp->propSetString(clip, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
    OfxParamSetHandle params;
    gEffect->getParamSet(effect, &params);
    OfxPropertySetHandle pp;
    gParam->paramDefine(params, kOfxParamTypeChoice, "profile", &pp);
    place(pp, "Film stock", "Film stock with its print or paper.", nullptr);
    int def = 0;
    for (int i = 0; i < static_cast<int>(gProfiles.size()); ++i) {
        gProp->propSetString(pp, kOfxParamPropChoiceOption, i, gProfiles[i].label.c_str());
        if (gProfiles[i].id == kDefaultProfile) def = i;
    }
    if (gProfiles.empty()) gProp->propSetString(pp, kOfxParamPropChoiceOption, 0, "(no valid profile)");
    gProp->propSetInt(pp, kOfxParamPropDefault, 0, def);
    OfxPropertySetHandle sp;
    gParam->paramDefine(params, kOfxParamTypeString, "profileId", &sp);
    gProp->propSetString(sp, kOfxPropLabel, 0, "Profile ID");
    gProp->propSetString(sp, kOfxParamPropDefault, 0, "");
    gProp->propSetInt(sp, kOfxParamPropSecret, 0, 1);
    gProp->propSetInt(sp, kOfxParamPropAnimates, 0, 0);
    OfxPropertySetHandle lp;
    gParam->paramDefine(params, kOfxParamTypeChoice, "shootingLight", &lp);
    gProp->propSetString(lp, kOfxPropLabel, 0, "Shooting light");
    for (int i = 0; i < 3; ++i) gProp->propSetString(lp, kOfxParamPropChoiceOption, i, kLightLabels[i]);
    gProp->propSetInt(lp, kOfxParamPropDefault, 0, 0);
    gProp->propSetInt(lp, kOfxParamPropSecret, 0, 1);
    define_bool(params, "greyBalance", "Grey balance", 0, "Off: the stock's own colour. On: neutral greys at every exposure.");
    define_bool(params, "labScan", "Lab scan look", 1, "On: colours of real scans of this film. Off: the manufacturer data only.");
    define_double(params, "exposure", "Exposure (stops)", 0.0, -5.0, 5.0, "Exposure of the film, in stops.");
    define_double(params, "pushPull", "Push / Pull (stops)", 0.0, -2.0, 2.0, "Push or pull development, in stops.");
    define_double(params, "impact", "Impact", 1.0, 0.0, 1.0, "Mix with the original image.");
    define_group(params, "grainGroup", "Grain", true);
    define_bool(params, "grainEnable", "Grain", 1, "Film grain on or off.", "grainGroup");
    define_double(params, "grainAmount", "Amount", 0.5, 0.0, 2.0, "Grain strength.", "grainGroup");
    define_double(params, "grainSize", "Size", 1.0, 0.2, 10.0, "Size of the grain clumps.", "grainGroup");
    define_choice(params, "grainRate", "Rate", kGrainRateLabels, 4, "How often the grain changes. Every frame is like real film.",
                  "grainGroup");
    OfxPropertySetHandle gp;
    gParam->paramDefine(params, kOfxParamTypeInteger, "grainSeed", &gp);
    place(gp, "Seed", "Picks a different grain pattern.", "grainGroup");
    gProp->propSetInt(gp, kOfxParamPropDefault, 0, 0);
    define_group(params, "glowGroup", "Halation & Bloom", true);
    define_double(params, "halationAmount", "Halation", 0.3, 0.0, 1.0, "Red glow around bright highlights.", "glowGroup");
    define_double(params, "halationRadius", "Halation radius", 6.0, 1.0, 30.0, "How far the halation spreads.", "glowGroup");
    define_double(params, "bloomAmount", "Bloom", 0.0, 0.0, 1.0, "Soft glow over bright areas.", "glowGroup");
    define_choice(params, "glowQuality", "Glow quality", kGlowQualityLabels, 2,
                  "Fast: halation and bloom at half resolution for smoother playback and less GPU memory. Full for the final render.",
                  "glowGroup");
    define_group(params, "printGroup", "Printer lights", false);
    define_double(params, "printExposure", "Print exposure", 0.0, -12.0, 12.0, "Print brightness, in printer points.", "printGroup");
    define_double(params, "printRed", "Red", 0.0, -12.0, 12.0, "Red printer light, in points.", "printGroup");
    define_double(params, "printGreen", "Green", 0.0, -12.0, 12.0, "Green printer light, in points.", "printGroup");
    define_double(params, "printBlue", "Blue", 0.0, -12.0, 12.0, "Blue printer light, in points.", "printGroup");
    define_group(params, "spaceGroup", "Colour spaces", false);
    define_choice(params, "inSpace", "Input", kSpaceLabels, 4, "Colour space of the incoming image.", "spaceGroup");
    define_choice(params, "outSpace", "Output", kSpaceLabels, 4, "Colour space of the result.", "spaceGroup");
    return kOfxStatOK;
}

double get_double(OfxImageEffectHandle effect, const char* name, OfxTime t) {
    OfxParamSetHandle params;
    gEffect->getParamSet(effect, &params);
    OfxParamHandle h;
    double v = 0.0;
    if (gParam->paramGetHandle(params, name, &h, nullptr) == kOfxStatOK) gParam->paramGetValueAtTime(h, t, &v);
    return v;
}

int get_choice(OfxImageEffectHandle effect, const char* name, OfxTime t) {
    OfxParamSetHandle params;
    gEffect->getParamSet(effect, &params);
    OfxParamHandle h;
    int v = 0;
    if (gParam->paramGetHandle(params, name, &h, nullptr) == kOfxStatOK) gParam->paramGetValueAtTime(h, t, &v);
    return v;
}

OfxParamHandle param_handle(OfxImageEffectHandle effect, const char* name) {
    OfxParamSetHandle params;
    gEffect->getParamSet(effect, &params);
    OfxParamHandle h = nullptr;
    gParam->paramGetHandle(params, name, &h, nullptr);
    return h;
}

int profile_by_id(const std::string& id) {
    for (int i = 0; i < static_cast<int>(gProfiles.size()); ++i)
        if (gProfiles[i].id == id) return i;
    return -1;
}

std::string stored_profile_id(OfxImageEffectHandle effect, OfxTime t) {
    OfxParamHandle h = param_handle(effect, "profileId");
    char* v = nullptr;
    if (h && gParam->paramGetValueAtTime(h, t, &v) == kOfxStatOK && v) return v;
    return {};
}

int resolve_profile(OfxImageEffectHandle effect, OfxTime t) {
    int by_id = profile_by_id(stored_profile_id(effect, t));
    return by_id >= 0 ? by_id : get_choice(effect, "profile", t);
}

OfxStatus is_identity(OfxImageEffectHandle effect, OfxPropertySetHandle in, OfxPropertySetHandle out) {
    OfxTime t = 0;
    gProp->propGetDouble(in, kOfxPropTime, 0, &t);
    if (get_double(effect, "impact", t) != 0.0) return kOfxStatReplyDefault;
    gProp->propSetString(out, kOfxPropName, 0, kOfxImageEffectSimpleSourceClipName);
    return kOfxStatOK;
}

struct ImageRef {
    OfxPropertySetHandle handle = nullptr;
    char* data = nullptr;
    int row_bytes = 0;
    OfxRectI bounds{};
    std::string depth, components, premult;
    ~ImageRef() {
        if (handle) gEffect->clipReleaseImage(handle);
    }
};

bool fetch(OfxImageClipHandle clip, OfxTime t, const OfxRectD* region, ImageRef& img) {
    if (gEffect->clipGetImage(clip, t, region, &img.handle) != kOfxStatOK) return false;
    void* ptr = nullptr;
    char* s = nullptr;
    gProp->propGetPointer(img.handle, kOfxImagePropData, 0, &ptr);
    gProp->propGetInt(img.handle, kOfxImagePropRowBytes, 0, &img.row_bytes);
    gProp->propGetIntN(img.handle, kOfxImagePropBounds, 4, &img.bounds.x1);
    gProp->propGetString(img.handle, kOfxImageEffectPropPixelDepth, 0, &s);
    img.depth = s ? s : "";
    gProp->propGetString(img.handle, kOfxImageEffectPropComponents, 0, &s);
    img.components = s ? s : "";
    gProp->propGetString(img.handle, kOfxImageEffectPropPreMultiplication, 0, &s);
    img.premult = s ? s : "";
    img.data = static_cast<char*>(ptr);
    return img.data != nullptr;
}

char* pixel_at(const ImageRef& img, int x, int y) {
    return img.data + static_cast<ptrdiff_t>(y - img.bounds.y1) * img.row_bytes +
           static_cast<ptrdiff_t>(x - img.bounds.x1) * 4 * sizeof(float);
}

OfxStatus render(OfxImageEffectHandle effect, OfxPropertySetHandle in) {
    OfxTime t = 0;
    gProp->propGetDouble(in, kOfxPropTime, 0, &t);
    int cuda_enabled = 0;
    gProp->propGetInt(in, kOfxImageEffectPropCudaEnabled, 0, &cuda_enabled);
    if (!cuda_enabled) {
        post_error(effect, "FIXER needs CUDA. Set Resolve GPU processing mode to CUDA.");
        return kOfxStatFailed;
    }
    int profile = resolve_profile(effect, t);
    if (profile < 0 || profile >= static_cast<int>(gProfiles.size())) {
        post_error(effect, "FIXER: no valid profile in the plugin Resources/profiles folder.");
        return kOfxStatFailed;
    }
    void* stream_ptr = nullptr;
    gProp->propGetPointer(in, kOfxImageEffectPropCudaStream, 0, &stream_ptr);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    OfxRectI win{};
    gProp->propGetIntN(in, kOfxImageEffectPropRenderWindow, 4, &win.x1);
    OfxImageClipHandle src_clip, dst_clip;
    gEffect->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &src_clip, nullptr);
    gEffect->clipGetHandle(effect, kOfxImageEffectOutputClipName, &dst_clip, nullptr);
    ImageRef src, dst;
    if (!fetch(dst_clip, t, nullptr, dst) || !fetch(src_clip, t, nullptr, src)) return kOfxStatFailed;
    if (src.depth != kOfxBitDepthFloat || dst.depth != kOfxBitDepthFloat || src.components != kOfxImageComponentRGBA ||
        dst.components != kOfxImageComponentRGBA)
        return kOfxStatErrUnsupported;
    win.x1 = std::max(win.x1, std::max(src.bounds.x1, dst.bounds.x1));
    win.y1 = std::max(win.y1, std::max(src.bounds.y1, dst.bounds.y1));
    win.x2 = std::min(win.x2, std::min(src.bounds.x2, dst.bounds.x2));
    win.y2 = std::min(win.y2, std::min(src.bounds.y2, dst.bounds.y2));
    int w = win.x2 - win.x1, h = win.y2 - win.y1;
    if (w <= 0 || h <= 0) return kOfxStatOK;
    if (src.row_bytes < w * 16 || dst.row_bytes < w * 16) {
        post_error(effect, "FIXER: unsupported image row layout.");
        return kOfxStatFailed;
    }
    Instance* inst = instance_of(effect);
    if (!inst) return kOfxStatFailed;
    std::lock_guard<std::mutex> guard(inst->lock);
    int device = 0;
    cudaGetDevice(&device);
    if (inst->device != device) {
        inst->luts.clear();
        inst->device = device;
    }
    const Profile& prof = gProfiles[profile];
    int light = get_choice(effect, "shootingLight", t);
    bool use_bal = get_choice(effect, "greyBalance", t) != 0 && (!prof.bal[0].empty() || !prof.bal[3].empty());
    bool use_alt = !use_bal && !prof.alt_light.empty() && light > 0 && light < 3 && prof.alt_light == kLightIds[light];
    bool use_scan = get_choice(effect, "labScan", t) != 0 && !prof.scan[0].empty();
    auto& luts = inst->luts[profile * 8 + (use_bal ? 2 : use_alt ? 1 : 0) + (use_scan ? 4 : 0)];
    if (!luts) {
        luts = std::make_unique<fe::LutSet>();
        const Profile& pr = prof;
        for (int s = 0; s < (pr.two_stage ? 4 : 3); ++s) {
            const std::vector<float>& data = (use_scan && s == 3) ? pr.scan[use_bal ? 1 : 0]
                                             : (use_bal && !pr.bal[s].empty()) ? pr.bal[s]
                                             : (use_alt && s < 3) ? pr.alt[s] : pr.slots[s];
            cudaError_t ue = luts->upload(s, data.data(), pr.size);
            if (ue != cudaSuccess) {
                luts.reset();
                post_error(effect, ue == cudaErrorMemoryAllocation
                                       ? "FIXER: not enough GPU memory to load the film stock (about 15 MB). Close other GPU programs or free GPU memory in Resolve."
                                       : "FIXER: GPU LUT upload failed.");
                return kOfxStatFailed;
            }
        }
        if (pr.two_stage) luts->set_print_domain(pr.print_lo, pr.print_hi);
        if (!pr.grain.empty()) luts->set_grain_table(pr.grain.data());
        luts->set_grain_mono(pr.grain_mono);
    }
    fe::RenderParams p{};
    p.in_space = get_choice(effect, "inSpace", t);
    p.out_space = get_choice(effect, "outSpace", t);
    p.exposure_ev = static_cast<float>(get_double(effect, "exposure", t));
    p.push_pull = static_cast<float>(get_double(effect, "pushPull", t));
    p.impact = static_cast<float>(get_double(effect, "impact", t));
    p.premultiplied = src.premult == kOfxImagePreMultiplied ? 1 : 0;
    double scale[2] = {1.0, 1.0};
    gProp->propGetDoubleN(in, kOfxImageEffectPropRenderScale, 2, scale);
    OfxRectD rod{};
    double frame_h = 1080.0;
    if (gEffect->clipGetRegionOfDefinition(src_clip, t, &rod) == kOfxStatOK && rod.y2 > rod.y1) frame_h = rod.y2 - rod.y1;
    double px_per_ref = frame_h / fe::kGrainRefHeight * scale[1];
    if (get_choice(effect, "grainEnable", t)) {
        p.grain_amount = static_cast<float>(get_double(effect, "grainAmount", t));
        p.grain_sigma_px = static_cast<float>(get_double(effect, "grainSize", t) * px_per_ref);
        p.grain_seed = static_cast<uint32_t>(get_choice(effect, "grainSeed", t));
        int step = kGrainRateSteps[std::clamp(get_choice(effect, "grainRate", t), 0, 3)];
        p.frame = step ? static_cast<int32_t>(std::floor(std::floor(t) / step)) : 0;
    }
    p.halation_amount = static_cast<float>(get_double(effect, "halationAmount", t));
    p.halation_radius_px = static_cast<float>(get_double(effect, "halationRadius", t) * px_per_ref);
    p.bloom_amount = static_cast<float>(get_double(effect, "bloomAmount", t));
    p.bloom_radius_px = static_cast<float>(fe::kBloomRadiusRef * px_per_ref);
    p.spatial_half = get_choice(effect, "glowQuality", t) != 0 ? 1 : 0;
    double print_exp = get_double(effect, "printExposure", t);
    const char* lights[3] = {"printRed", "printGreen", "printBlue"};
    for (int c = 0; c < 3; ++c)
        p.print_log_offset[c] = static_cast<float>(fe::kPrinterPoint * (print_exp + get_double(effect, lights[c], t)));
    fe::Rect src_bounds{src.bounds.x1, src.bounds.y1, src.bounds.x2, src.bounds.y2};
    fe::Rect dst_bounds{dst.bounds.x1, dst.bounds.y1, dst.bounds.x2, dst.bounds.y2};
    cudaError_t e = fe::render_region(fe::Image{src.data, static_cast<size_t>(src.row_bytes), src_bounds},
                                      reinterpret_cast<float*>(dst.data), static_cast<size_t>(dst.row_bytes), dst_bounds,
                                      fe::Rect{win.x1, win.y1, win.x2, win.y2}, p, *luts, stream);
    if (e == cudaSuccess && !stream) e = cudaStreamSynchronize(nullptr);
    if (e == cudaErrorMemoryAllocation) {
        bool glow = p.halation_amount > 0.0f || p.bloom_amount > 0.0f;
        double mb = static_cast<double>(w) * h * 16.0 * (glow ? (p.spatial_half ? 2.0 : 4.0) : 1.0) / (1024.0 * 1024.0);
        char msg[320];
        std::snprintf(msg, sizeof msg,
                      "FIXER: not enough GPU memory for this frame (about %.0f MB needed). %s lower the timeline resolution, or close other GPU programs.",
                      mb, glow && !p.spatial_half ? "Set Glow quality to Fast," : "Turn halation and bloom down,");
        post_error(effect, msg);
        return kOfxStatFailed;
    }
    if (e != cudaSuccess) {
        post_error(effect, cudaGetErrorString(e));
        return kOfxStatFailed;
    }
    return kOfxStatOK;
}

OfxStatus create_instance(OfxImageEffectHandle effect) {
    OfxPropertySetHandle props;
    gEffect->getPropertySet(effect, &props);
    gProp->propSetPointer(props, kOfxPropInstanceData, 0, new Instance());
    int by_id = profile_by_id(stored_profile_id(effect, 0.0));
    OfxParamHandle choice = param_handle(effect, "profile");
    if (by_id >= 0 && choice && by_id != get_choice(effect, "profile", 0.0)) gParam->paramSetValue(choice, by_id);
    return kOfxStatOK;
}

OfxStatus instance_changed(OfxImageEffectHandle effect, OfxPropertySetHandle in) {
    char* type = nullptr;
    char* name = nullptr;
    char* reason = nullptr;
    gProp->propGetString(in, kOfxPropType, 0, &type);
    gProp->propGetString(in, kOfxPropName, 0, &name);
    gProp->propGetString(in, kOfxPropChangeReason, 0, &reason);
    if (!type || !name || !reason || std::strcmp(type, kOfxTypeParameter) || std::strcmp(name, "profile") ||
        !std::strcmp(reason, kOfxChangePluginEdited))
        return kOfxStatReplyDefault;
    OfxTime t = 0;
    gProp->propGetDouble(in, kOfxPropTime, 0, &t);
    int idx = get_choice(effect, "profile", t);
    OfxParamHandle h = param_handle(effect, "profileId");
    if (h && idx >= 0 && idx < static_cast<int>(gProfiles.size())) gParam->paramSetValue(h, gProfiles[idx].id.c_str());
    return kOfxStatOK;
}

OfxStatus destroy_instance(OfxImageEffectHandle effect) {
    delete instance_of(effect);
    OfxPropertySetHandle props;
    gEffect->getPropertySet(effect, &props);
    gProp->propSetPointer(props, kOfxPropInstanceData, 0, nullptr);
    return kOfxStatOK;
}

OfxStatus on_load() {
    if (!gHost) return kOfxStatErrMissingHostFeature;
    gProp = static_cast<OfxPropertySuiteV1*>(const_cast<void*>(gHost->fetchSuite(gHost->host, kOfxPropertySuite, 1)));
    gEffect = static_cast<OfxImageEffectSuiteV1*>(const_cast<void*>(gHost->fetchSuite(gHost->host, kOfxImageEffectSuite, 1)));
    gParam = static_cast<OfxParameterSuiteV1*>(const_cast<void*>(gHost->fetchSuite(gHost->host, kOfxParameterSuite, 1)));
    gMessage = static_cast<OfxMessageSuiteV1*>(const_cast<void*>(gHost->fetchSuite(gHost->host, kOfxMessageSuite, 1)));
    if (!gProp || !gEffect || !gParam) return kOfxStatErrMissingHostFeature;
    load_profiles();
    return kOfxStatOK;
}

OfxStatus main_entry(const char* action, const void* handle, OfxPropertySetHandle in, OfxPropertySetHandle out) {
    auto effect = static_cast<OfxImageEffectHandle>(const_cast<void*>(handle));
    if (!std::strcmp(action, kOfxActionLoad)) return on_load();
    if (!std::strcmp(action, kOfxActionDescribe)) return describe(effect);
    if (!std::strcmp(action, kOfxImageEffectActionDescribeInContext)) return describe_in_context(effect);
    if (!std::strcmp(action, kOfxActionCreateInstance)) return create_instance(effect);
    if (!std::strcmp(action, kOfxActionDestroyInstance)) return destroy_instance(effect);
    if (!std::strcmp(action, kOfxImageEffectActionIsIdentity)) return is_identity(effect, in, out);
    if (!std::strcmp(action, kOfxImageEffectActionRender)) return render(effect, in);
    if (!std::strcmp(action, kOfxActionInstanceChanged)) return instance_changed(effect, in);
    return kOfxStatReplyDefault;
}

void set_host(OfxHost* host) { gHost = host; }

OfxPlugin gPlugin = {kOfxImageEffectPluginApi, 1, kPluginId, 0, 1, set_host, main_entry};

}

OfxExport int OfxGetNumberOfPlugins(void) { return 1; }

OfxExport OfxPlugin* OfxGetPlugin(int nth) { return nth == 0 ? &gPlugin : nullptr; }
