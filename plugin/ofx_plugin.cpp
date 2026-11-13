#define NOMINMAX
#include <windows.h>
#include <objbase.h>
#include <wincodec.h>
#include <commctrl.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
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
#include "ofxProgress.h"
#include "ofxProperty.h"
#include "match.h"

#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "comctl32.lib")

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
OfxProgressSuiteV1* gProgress = nullptr;

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
constexpr const char* kGlowQualityLabels[] = {"Full", "Fast"};
// Resolve 21 exposes this extension through its host descriptor.
constexpr const char* kButtonRowType = "OfxParamTypeButtonRow";
constexpr const char* kButtonRowSupport = "OfxParamHostPropSupportsButtonRowAnimation";
bool gButtonRows = false;
constexpr const char* kButtonChoices[] = {"glowQuality", "matchSource", "matchMode", "matchShow"};
constexpr const char* kLightIds[] = {"", "daylight", "tungsten"};
constexpr const char* kMatchSourceLabels[] = {"Film + match", "Match only"};
constexpr const char* kMatchModeLabels[] = {"Simple", "Detailed"};
constexpr const char* kMatchShowLabels[] = {"Result", "Before match"};

std::vector<Profile> gProfiles;
std::vector<std::wstring> gSavedLooks;

struct Instance {
    std::mutex lock;
    std::map<int, std::unique_ptr<fe::LutSet>> luts;
    int device = -1;
    std::string match_text;
    double match_s = -1.0, match_p = -1.0;
    bool match_ok = false, match_only = false, order_warned = false;
    std::vector<float> match_lut;
    unsigned match_gen = 0, match_set_gen = 0;
    std::unique_ptr<fe::LutSet> match_set, match_prev;
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

OfxPropertySetHandle define_choice(OfxParamSetHandle params, const char* name, const char* label, const char* const* options, int count,
                   const char* hint, const char* parent = nullptr) {
    OfxPropertySetHandle p;
    gParam->paramDefine(params, kOfxParamTypeChoice, name, &p);
    place(p, label, hint, parent);
    for (int i = 0; i < count; ++i) gProp->propSetString(p, kOfxParamPropChoiceOption, i, options[i]);
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 0);
    return p;
}

std::string choice_button_name(const char* name, int index) {
    return std::string(name) + "Option" + std::to_string(index);
}

std::string choice_row_name(const char* name) { return std::string(name) + "Row"; }

int choice_row_index(const char* name, int index) {
    // Display Fast first without changing the saved Full=0 / Fast=1 values.
    return !std::strcmp(name, "glowQuality") ? 1 - index : index;
}

std::string choice_button_label(const char* label, const char* option, bool selected) {
    return std::string(selected ? "\xE2\x97\x8F " : "") + label + ": " + option;
}

void define_button_choice(OfxParamSetHandle params, const char* name, const char* label,
                          const char* const* options, int count, const char* hint, const char* parent) {
    // Keep the original choice ID/type so saved projects and animation remain compatible.
    OfxPropertySetHandle choice = define_choice(params, name, label, options, count, hint, parent);
    gProp->propSetInt(choice, kOfxParamPropSecret, 0, 1);
    if (gButtonRows) {
        OfxPropertySetHandle row;
        if (gParam->paramDefine(params, kButtonRowType, choice_row_name(name).c_str(), &row) != kOfxStatOK)
            throw std::runtime_error("Host advertised ButtonRow but rejected it");
        place(row, (std::string(label) + ":").c_str(), hint, parent);
        // ButtonRow uses string enum values and eight icon slots per option.
        // Empty icons keep the host's text buttons instead of image-only controls.
        for (int i = 0; i < count; ++i) {
            int value = choice_row_index(name, i);
            gProp->propSetString(row, kOfxParamPropChoiceOption, i, options[value]);
            gProp->propSetString(row, "OfxParamPropChoiceEnum", i, std::to_string(value).c_str());
            for (int icon = 0; icon < 8; ++icon)
                gProp->propSetString(row, "OfxParamPropChoiceIcon", i * 8 + icon, "");
        }
        gProp->propSetString(row, kOfxParamPropDefault, 0, "0");
        gProp->propSetInt(row, kOfxParamPropAnimates, 0, 0);
        gProp->propSetInt(row, kOfxParamPropPersistant, 0, 0);
        gProp->propSetInt(row, kOfxParamPropEvaluateOnChange, 0, 0);
    }
    for (int i = 0; i < count; ++i) {
        OfxPropertySetHandle button;
        gParam->paramDefine(params, kOfxParamTypePushButton, choice_button_name(name, i).c_str(), &button);
        place(button, choice_button_label(label, options[i], i == 0).c_str(), hint, parent);
        gProp->propSetInt(button, kOfxParamPropEvaluateOnChange, 0, 0);
        gProp->propSetInt(button, kOfxParamPropSecret, 0, gButtonRows ? 1 : 0);
    }
}

void define_string(OfxParamSetHandle params, const char* name, const char* label, const char* hint, bool file) {
    OfxPropertySetHandle p;
    gParam->paramDefine(params, kOfxParamTypeString, name, &p);
    place(p, label, hint, "matchGroup");
    gProp->propSetString(p, kOfxParamPropDefault, 0, "");
    gProp->propSetInt(p, kOfxParamPropAnimates, 0, 0);
    if (file) {
        gProp->propSetString(p, kOfxParamPropStringMode, 0, kOfxParamStringIsFilePath);
        gProp->propSetInt(p, kOfxParamPropStringFilePathExists, 0, 1);
    }
}

void define_saved_looks(OfxParamSetHandle params);

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
    define_button_choice(params, "glowQuality", "Glow", kGlowQualityLabels, 2,
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
    define_group(params, "matchGroup", "Reference match", false);
    define_string(params, "matchFile", "Reference", "JPEG or PNG reference image, read as sRGB.", true);
    define_button_choice(params, "matchSource", "Order", kMatchSourceLabels, 2,
                  "Film + match: the match goes on top of the film. Match only: no film, only the match.", "matchGroup");
    define_button_choice(params, "matchMode", "Mode", kMatchModeLabels, 2,
                  "Simple: tone curve, saturation and three tints. Detailed is not available yet and works as Simple.", "matchGroup");
    OfxPropertySetHandle bp;
    gParam->paramDefine(params, kOfxParamTypePushButton, "matchCreate", &bp);
    place(bp, "Create grade", "Match the current frame to the reference image.", "matchGroup");
    define_double(params, "matchStrength", "Strength", 0.7, 0.0, 1.0, "How much of the match is applied.", "matchGroup");
    define_double(params, "matchProtectSkin", "Protect skin tones", 0.5, 0.0, 1.0,
                  "Applies less of the match to skin-coloured tones. Objects of a similar colour are protected too.", "matchGroup");
    define_button_choice(params, "matchShow", "Show", kMatchShowLabels, 2, "Before match: the image without the match, to compare.",
                  "matchGroup");
    OfxPropertySetHandle mp;
    gParam->paramDefine(params, kOfxParamTypeString, "matchState", &mp);
    gProp->propSetString(mp, kOfxPropLabel, 0, "Match state");
    gProp->propSetString(mp, kOfxParamPropDefault, 0, "");
    gProp->propSetInt(mp, kOfxParamPropSecret, 0, 1);
    gProp->propSetInt(mp, kOfxParamPropAnimates, 0, 0);
    define_string(params, "lookName", "Look name", "Name of the look to save.", false);
    OfxPropertySetHandle sbp;
    gParam->paramDefine(params, kOfxParamTypePushButton, "lookSave", &sbp);
    place(sbp, "Save look", "Saves the look as .json and .cube (match only, Rec.709 gamma 2.4) in %APPDATA%\\FIXER\\looks.",
          "matchGroup");
    define_string(params, "lookLoad", "Load look", "Pick a look .json saved by FIXER.", true);
    define_saved_looks(params);
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

void sync_choice_buttons(OfxImageEffectHandle effect, OfxTime time) {
    OfxParamSetHandle params;
    gEffect->getParamSet(effect, &params);
    for (const char* name : kButtonChoices) {
        OfxParamHandle choice;
        OfxPropertySetHandle props;
        if (gParam->paramGetHandle(params, name, &choice, &props) != kOfxStatOK) continue;
        int count = 0, selected = get_choice(effect, name, time);
        if (gButtonRows) {
            OfxParamHandle row;
            char* current = nullptr;
            if (gParam->paramGetHandle(params, choice_row_name(name).c_str(), &row, nullptr) == kOfxStatOK) {
                gParam->paramGetValueAtTime(row, time, &current);
                const std::string wanted = std::to_string(selected);
                if (!current || wanted != current) gParam->paramSetValue(row, wanted.c_str());
            }
        }
        char* label = nullptr;
        gProp->propGetDimension(props, kOfxParamPropChoiceOption, &count);
        gProp->propGetString(props, kOfxPropLabel, 0, &label);
        for (int i = 0; i < count; ++i) {
            char* option = nullptr;
            OfxParamHandle button;
            OfxPropertySetHandle bp;
            gProp->propGetString(props, kOfxParamPropChoiceOption, i, &option);
            if (label && option && gParam->paramGetHandle(params, choice_button_name(name, i).c_str(), &button, &bp) == kOfxStatOK)
                gProp->propSetString(bp, kOfxPropLabel, 0, choice_button_label(label, option, i == selected).c_str());
        }
    }
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

OfxStatus fail_msg(OfxImageEffectHandle effect, const char* text) {
    post_error(effect, text);
    return kOfxStatFailed;
}

std::string get_string(OfxImageEffectHandle effect, const char* name, OfxTime t) {
    OfxParamHandle h = param_handle(effect, name);
    char* v = nullptr;
    if (h && gParam->paramGetValueAtTime(h, t, &v) == kOfxStatOK && v) return v;
    return {};
}

std::wstring widen(const std::string& s) {
    if (s.empty() || s.size() > static_cast<size_t>(INT_MAX)) return {};
    const UINT pages[2] = {CP_UTF8, CP_ACP};
    for (UINT cp : pages) {
        DWORD flags = cp == CP_UTF8 ? MB_ERR_INVALID_CHARS : 0;
        int n = MultiByteToWideChar(cp, flags, s.data(), static_cast<int>(s.size()), nullptr, 0);
        if (n <= 0) continue;
        std::wstring w(static_cast<size_t>(n), L'\0');
        if (MultiByteToWideChar(cp, flags, s.data(), static_cast<int>(s.size()), w.data(), n) == n) return w;
    }
    return {};
}

std::string narrow(const std::wstring& w) {
    if (w.empty() || w.size() > static_cast<size_t>(INT_MAX)) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    if (n <= 0) return {};
    std::string s(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

double now_ms() {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1000.0 * static_cast<double>(c.QuadPart) / static_cast<double>(f.QuadPart);
}

void log_timing(OfxImageEffectHandle effect, const char* text) {
    if (gMessage && GetEnvironmentVariableW(L"FIXER_TIMING", nullptr, 0) > 0) gMessage->message(effect, kOfxMessageLog, nullptr, "%s", text);
}

const char* lut_error(cudaError_t e) {
    return e == cudaErrorMemoryAllocation
               ? "FIXER: not enough GPU memory to load the film stock (about 15 MB). Close other GPU programs or free GPU memory in Resolve."
               : "FIXER: GPU LUT upload failed.";
}

int film_key(OfxImageEffectHandle effect, int profile, OfxTime t) {
    const Profile& prof = gProfiles[profile];
    int light = get_choice(effect, "shootingLight", t);
    bool use_bal = get_choice(effect, "greyBalance", t) != 0 && (!prof.bal[0].empty() || !prof.bal[3].empty());
    bool use_alt = !use_bal && !prof.alt_light.empty() && light > 0 && light < 3 && prof.alt_light == kLightIds[light];
    bool use_scan = get_choice(effect, "labScan", t) != 0 && !prof.scan[0].empty();
    return profile * 8 + (use_bal ? 2 : use_alt ? 1 : 0) + (use_scan ? 4 : 0);
}

cudaError_t fill_film(int key, fe::LutSet& luts) {
    const Profile& pr = gProfiles[key / 8];
    bool use_alt = (key & 1) != 0, use_bal = (key & 2) != 0, use_scan = (key & 4) != 0;
    for (int s = 0; s < (pr.two_stage ? 4 : 3); ++s) {
        const std::vector<float>& data = (use_scan && s == 3) ? pr.scan[use_bal ? 1 : 0]
                                         : (use_bal && !pr.bal[s].empty()) ? pr.bal[s]
                                         : (use_alt && s < 3) ? pr.alt[s] : pr.slots[s];
        cudaError_t ue = luts.upload(s, data.data(), pr.size);
        if (ue != cudaSuccess) return ue;
    }
    if (pr.two_stage) luts.set_print_domain(pr.print_lo, pr.print_hi);
    if (!pr.grain.empty()) luts.set_grain_table(pr.grain.data());
    luts.set_grain_mono(pr.grain_mono);
    return cudaSuccess;
}

fe::LutSet* film_luts(OfxImageEffectHandle effect, Instance* inst, int profile, OfxTime t) {
    int device = 0;
    cudaGetDevice(&device);
    if (inst->device != device) {
        inst->luts.clear();
        inst->match_set.reset();
        inst->match_prev.reset();
        inst->device = device;
    }
    int key = film_key(effect, profile, t);
    auto& luts = inst->luts[key];
    if (!luts) {
        luts = std::make_unique<fe::LutSet>();
        cudaError_t ue = fill_film(key, *luts);
        if (ue != cudaSuccess) {
            luts.reset();
            post_error(effect, lut_error(ue));
            return nullptr;
        }
    }
    return luts.get();
}

fe::RenderParams base_params(OfxImageEffectHandle effect, OfxTime t) {
    fe::RenderParams p{};
    p.in_space = get_choice(effect, "inSpace", t);
    p.out_space = get_choice(effect, "outSpace", t);
    p.exposure_ev = static_cast<float>(get_double(effect, "exposure", t));
    p.push_pull = static_cast<float>(get_double(effect, "pushPull", t));
    double impact = get_double(effect, "impact", t);
    p.impact = static_cast<float>(std::isfinite(impact) ? std::clamp(impact, 0.0, 1.0) : 1.0);
    double print_exp = get_double(effect, "printExposure", t);
    const char* lights[3] = {"printRed", "printGreen", "printBlue"};
    for (int c = 0; c < 3; ++c)
        p.print_log_offset[c] = static_cast<float>(fe::kPrinterPoint * (print_exp + get_double(effect, lights[c], t)));
    return p;
}

void cache_match(Instance* inst, const std::string& text, double s, double pr, OfxImageEffectHandle effect) {
    fxm::State st;
    std::string why;
    bool new_text = text != inst->match_text;
    inst->match_text = text;
    inst->match_s = s;
    inst->match_p = pr;
    inst->match_ok = !text.empty() && fxm::parse_state(text, st, &why);
    if (new_text) inst->order_warned = false;
    if (new_text && !text.empty() && !inst->match_ok) {
        std::string msg = "FIXER: the stored reference match is invalid (" + why + "). Press Create grade again.";
        if (gMessage) gMessage->message(effect, kOfxMessageLog, nullptr, "%s", msg.c_str());
    }
    if (inst->match_ok) {
        double t0 = now_ms();
        inst->match_only = st.source == "only";
        inst->match_lut = fxm::bake_lut(st, s, pr);
        char msg[96];
        std::snprintf(msg, sizeof msg, "FIXER timing: match LUT bake %.1f ms", now_ms() - t0);
        log_timing(effect, msg);
    }
    ++inst->match_gen;
}

bool apply_match(OfxImageEffectHandle effect, Instance* inst, fe::LutSet& luts, OfxTime t, fe::RenderParams& p, cudaStream_t stream) {
    double s = fxm::clip(get_double(effect, "matchStrength", t), 0.0, 1.0);
    double pr = fxm::clip(get_double(effect, "matchProtectSkin", t), 0.0, 1.0);
    if (!(s > 0.0) || get_choice(effect, "matchShow", t) != 0) return true;
    std::string text = get_string(effect, "matchState", t);
    if (text != inst->match_text || s != inst->match_s || pr != inst->match_p) cache_match(inst, text, s, pr, effect);
    if (!inst->match_ok) return true;
    if (!inst->match_set || inst->match_set_gen != inst->match_gen) {
        double t0 = now_ms();
        auto fresh = std::make_unique<fe::LutSet>();
        cudaError_t e = fresh->upload(fe::LutSet::kMatchSlot, inst->match_lut.data(), fxm::kLutSize);
        if (e != cudaSuccess) {
            post_error(effect, e == cudaErrorMemoryAllocation ? "FIXER: not enough GPU memory to load the reference match."
                                                              : "FIXER: GPU upload of the reference match failed.");
            return false;
        }
        if (inst->match_prev) {
            cudaStreamSynchronize(stream);
            inst->match_prev.reset();
        }
        inst->match_prev = std::move(inst->match_set);
        inst->match_set = std::move(fresh);
        inst->match_set_gen = inst->match_gen;
        char msg[96];
        std::snprintf(msg, sizeof msg, "FIXER timing: match LUT upload %.1f ms", now_ms() - t0);
        log_timing(effect, msg);
    }
    luts.borrow_match(*inst->match_set);
    if ((get_choice(effect, "matchSource", t) == 1) != inst->match_only && !inst->order_warned) {
        inst->order_warned = true;
        if (gMessage) gMessage->message(effect, kOfxMessageLog, nullptr, "%s", "FIXER: Order changed: press Create grade again.");
    }
    if (inst->match_only) p.reserved2[0] |= fe::LutSet::kFilmBypass;
    p.reserved = fxm::kLutSize;
    p.reserved2[0] |= fe::LutSet::kMatchOn;
    return true;
}

int resolve_profile(OfxImageEffectHandle effect, OfxTime t) {
    std::string id = stored_profile_id(effect, t);
    int by_id = profile_by_id(id.empty() ? kDefaultProfile : id);
    return by_id >= 0 ? by_id : get_choice(effect, "profile", t);
}

OfxStatus is_identity(OfxImageEffectHandle effect, OfxPropertySetHandle in, OfxPropertySetHandle out) {
    OfxTime t = 0;
    gProp->propGetDouble(in, kOfxPropTime, 0, &t);
    double impact = get_double(effect, "impact", t);
    if (!std::isfinite(impact) || impact > 0.0) return kOfxStatReplyDefault;
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
    fe::LutSet* luts = film_luts(effect, inst, profile, t);
    if (!luts) return kOfxStatFailed;
    fe::RenderParams p = base_params(effect, t);
    if (!apply_match(effect, inst, *luts, t, p, stream)) return kOfxStatFailed;
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
    std::string id = stored_profile_id(effect, 0.0);
    int by_id = profile_by_id(id.empty() ? kDefaultProfile : id);
    OfxParamHandle choice = param_handle(effect, "profile");
    if (by_id >= 0 && choice && by_id != get_choice(effect, "profile", 0.0)) gParam->paramSetValue(choice, by_id);
    sync_choice_buttons(effect, 0.0);
    return kOfxStatOK;
}

HMODULE self_module() {
    HMODULE m = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&self_module), &m);
    return m;
}

struct Progress {
    OfxImageEffectHandle effect;
    bool host = false;
    HWND wnd = nullptr, label = nullptr, bar = nullptr;
    double t0 = 0.0;

    Progress(OfxImageEffectHandle e, const char* title) : effect(e), t0(now_ms()) {
        if (gProgress && gProgress->progressStart && gProgress->progressStart(effect, title) == kOfxStatOK) {
            host = true;
            return;
        }
        wchar_t off[4] = {};
        if (GetEnvironmentVariableW(L"FIXER_NO_PROGRESS", off, 4) == 1 && off[0] == L'1') return;
        open(title);
    }
    ~Progress() { close(); }
    Progress(const Progress&) = delete;
    Progress& operator=(const Progress&) = delete;

    void open(const char* title) {
        INITCOMMONCONTROLSEX icc{sizeof icc, ICC_PROGRESS_CLASS};
        InitCommonControlsEx(&icc);
        HINSTANCE hi = self_module();
        WNDCLASSW wc{};
        wc.lpfnWndProc = DefWindowProcW;
        wc.hInstance = hi;
        wc.hCursor = LoadCursorA(nullptr, IDC_WAIT);
        wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
        wc.lpszClassName = L"FIXERProgress";
        RegisterClassW(&wc);
        HWND owner = GetForegroundWindow();
        DWORD pid = 0;
        if (owner) GetWindowThreadProcessId(owner, &pid);
        RECT r{};
        if (!owner || !GetWindowRect(owner, &r) || r.right <= r.left) SystemParametersInfoW(SPI_GETWORKAREA, 0, &r, 0);
        int w = 380, h = 112;
        wnd = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW, L"FIXERProgress", widen(title).c_str(), WS_POPUP | WS_CAPTION,
                              (r.left + r.right - w) / 2, (r.top + r.bottom - h) / 2, w, h,
                              pid == GetCurrentProcessId() ? owner : nullptr, nullptr, hi, nullptr);
        if (!wnd) return;
        HGDIOBJ font = GetStockObject(DEFAULT_GUI_FONT);
        label = CreateWindowExW(0, L"STATIC", L"", WS_CHILD | WS_VISIBLE, 16, 12, w - 40, 20, wnd, nullptr, hi, nullptr);
        bar = CreateWindowExW(0, PROGRESS_CLASSW, nullptr, WS_CHILD | WS_VISIBLE | PBS_SMOOTH, 16, 40, w - 40, 18, wnd, nullptr, hi, nullptr);
        if (label) SendMessageW(label, WM_SETFONT, reinterpret_cast<WPARAM>(font), FALSE);
        if (bar) SendMessageW(bar, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
        ShowWindow(wnd, SW_SHOWNOACTIVATE);
        pump();
    }

    void pump() {
        if (!wnd) return;
        RedrawWindow(wnd, nullptr, nullptr, RDW_INVALIDATE | RDW_UPDATENOW | RDW_ALLCHILDREN);
        MSG m;
        const HWND windows[3] = {wnd, label, bar};
        for (HWND h : windows)
            while (h && PeekMessageW(&m, h, 0, 0, PM_REMOVE)) {
                TranslateMessage(&m);
                DispatchMessageW(&m);
            }
    }

    void step(const char* name, int pct) {
        if (host) {
            if (gProgress->progressUpdate) gProgress->progressUpdate(effect, pct / 100.0);
            return;
        }
        if (!wnd) return;
        if (label) SetWindowTextW(label, widen(name).c_str());
        if (bar) SendMessageW(bar, PBM_SETPOS, static_cast<WPARAM>(pct), 0);
        pump();
    }

    void close() {
        if (host) {
            host = false;
            if (gProgress->progressEnd) gProgress->progressEnd(effect);
            return;
        }
        if (!wnd) return;
        while (now_ms() - t0 < 500.0) {
            pump();
            Sleep(10);
        }
        DestroyWindow(wnd);
        wnd = label = bar = nullptr;
    }
};

struct EditBlock {
    OfxParamSetHandle params = nullptr;
    EditBlock(OfxImageEffectHandle effect, const char* label) {
        gEffect->getParamSet(effect, &params);
        if (gParam->paramEditBegin) gParam->paramEditBegin(params, label);
    }
    ~EditBlock() {
        if (gParam->paramEditEnd) gParam->paramEditEnd(params);
    }
    EditBlock(const EditBlock&) = delete;
    EditBlock& operator=(const EditBlock&) = delete;
};

struct DeviceScope {
    int prev = -1;
    ~DeviceScope() {
        if (prev >= 0) cudaSetDevice(prev);
    }
};

constexpr UINT kRefMaxSide = 30000, kRefScaled = 1024;

bool image_complete(const std::wstring& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    static const std::string png("\x89PNG\r\n\x1a\n", 8), iend("IEND\xAE\x42\x60\x82", 8);
    if (data.compare(0, 8, png) == 0) return data.find(iend, 8) != std::string::npos;
    if (data.compare(0, 2, "\xFF\xD8") == 0) return data.find("\xFF\xD9", 2) != std::string::npos;
    return true;
}

int read_reference(const std::wstring& path, std::vector<uint8_t>& px, int& w, int& h) {
    if (!image_complete(path)) return 1;
    HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    IWICImagingFactory* factory = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICBitmapScaler* scaler = nullptr;
    IWICFormatConverter* conv = nullptr;
    UINT fw = 0, fh = 0;
    int status = 1;
    bool ok = !path.empty() &&
              SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory))) &&
              SUCCEEDED(factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &decoder)) &&
              SUCCEEDED(decoder->GetFrame(0, &frame)) && SUCCEEDED(frame->GetSize(&fw, &fh)) && fw > 0 && fh > 0;
    if (ok && (fw > kRefMaxSide || fh > kRefMaxSide)) {
        ok = false;
        status = 2;
    }
    if (ok) {
        UINT longest = std::max(fw, fh), sw = fw, sh = fh;
        if (longest > kRefScaled) {
            sw = std::max<UINT>(1, static_cast<UINT>(std::llround(static_cast<double>(fw) * kRefScaled / longest)));
            sh = std::max<UINT>(1, static_cast<UINT>(std::llround(static_cast<double>(fh) * kRefScaled / longest)));
        }
        ok = SUCCEEDED(factory->CreateFormatConverter(&conv)) &&
             SUCCEEDED(conv->Initialize(frame, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0.0,
                                        WICBitmapPaletteTypeCustom));
        IWICBitmapSource* src = conv;
        if (ok && (sw != fw || sh != fh)) {
            ok = SUCCEEDED(factory->CreateBitmapScaler(&scaler)) &&
                 SUCCEEDED(scaler->Initialize(conv, sw, sh, WICBitmapInterpolationModeFant));
            src = scaler;
        }
        UINT cw = 0, ch = 0;
        ok = ok && SUCCEEDED(src->GetSize(&cw, &ch)) && cw == sw && ch == sh;
        size_t stride = static_cast<size_t>(sw) * 4;
        size_t bytes = stride * sh;
        ok = ok && stride / 4 == sw && bytes / stride == sh && bytes <= UINT_MAX;
        if (ok) {
            px.resize(bytes);
            ok = SUCCEEDED(src->CopyPixels(nullptr, static_cast<UINT>(stride), static_cast<UINT>(bytes), px.data()));
            w = static_cast<int>(sw);
            h = static_cast<int>(sh);
        }
        if (ok) status = 0;
    }
    if (conv) conv->Release();
    if (scaler) scaler->Release();
    if (frame) frame->Release();
    if (decoder) decoder->Release();
    if (factory) factory->Release();
    if (SUCCEEDED(init)) CoUninitialize();
    return status;
}

bool write_file(const std::wstring& path, const std::string& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f.write(data.data(), static_cast<std::streamsize>(data.size()))) return false;
    f.close();
    return !f.fail();
}

std::wstring looks_dir() {
    wchar_t appdata[MAX_PATH * 4];
    DWORD n = GetEnvironmentVariableW(L"APPDATA", appdata, static_cast<DWORD>(std::size(appdata)));
    if (n == 0 || n >= std::size(appdata)) return {};
    return std::wstring(appdata, n) + L"\\FIXER\\looks";
}

void define_saved_looks(OfxParamSetHandle params) {
    gSavedLooks.clear();
    std::vector<std::pair<std::string, std::wstring>> looks;
    std::wstring dir = looks_dir();
    WIN32_FIND_DATAW fd;
    HANDLE h = dir.empty() ? INVALID_HANDLE_VALUE : FindFirstFileW((dir + L"\\*.json").c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            std::wstring file = fd.cFileName;
            if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || file.size() <= 5 || _wcsicmp(file.c_str() + file.size() - 5, L".json"))
                continue;
            std::string label = narrow(file.substr(0, file.size() - 5));
            if (!label.empty()) looks.emplace_back(label, dir + L"\\" + file);
        } while (FindNextFileW(h, &fd));
        FindClose(h);
    }
    std::sort(looks.begin(), looks.end(), [](const auto& a, const auto& b) { return natural_less(a.first, b.first); });
    OfxPropertySetHandle p;
    gParam->paramDefine(params, kOfxParamTypeChoice, "lookSaved", &p);
    place(p, "Saved looks", "Looks saved in %APPDATA%\\FIXER\\looks. New saves appear after restarting Resolve.", "matchGroup");
    gProp->propSetString(p, kOfxParamPropChoiceOption, 0, "(none)");
    for (size_t i = 0; i < looks.size(); ++i) {
        gProp->propSetString(p, kOfxParamPropChoiceOption, static_cast<int>(i + 1), looks[i].first.c_str());
        gSavedLooks.push_back(looks[i].second);
    }
    gProp->propSetInt(p, kOfxParamPropDefault, 0, 0);
    gProp->propSetInt(p, kOfxParamPropAnimates, 0, 0);
}

OfxStatus create_grade(OfxImageEffectHandle effect, OfxTime t) {
    std::string ref_path = get_string(effect, "matchFile", t);
    if (ref_path.empty()) return fail_msg(effect, "FIXER: choose a reference image first (Reference match > Reference).");
    int profile = resolve_profile(effect, t);
    if (profile < 0 || profile >= static_cast<int>(gProfiles.size()))
        return fail_msg(effect, "FIXER: no valid profile in the plugin Resources/profiles folder.");
    Instance* inst = instance_of(effect);
    if (!inst) return kOfxStatFailed;
    OfxImageClipHandle src_clip = nullptr;
    gEffect->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &src_clip, nullptr);
    Progress prog(effect, "FIXER: Create grade");
    auto fail = [&](const char* text) {
        prog.close();
        return fail_msg(effect, text);
    };
    double ms[7] = {}, t0 = now_ms();
    prog.step("Reading frame", 0);
    int dw = 0, dh = 0;
    std::vector<float> host;
    {
        ImageRef img;
        if (!src_clip || !fetch(src_clip, t, nullptr, img) || img.depth != kOfxBitDepthFloat || img.components != kOfxImageComponentRGBA)
            return fail("FIXER: cannot read the current frame for the reference match.");
        cudaPointerAttributes attr{};
        bool on_device = cudaPointerGetAttributes(&attr, img.data) == cudaSuccess && attr.type == cudaMemoryTypeDevice;
        cudaGetLastError();
        if (on_device) return fail("FIXER: the host gave the frame in GPU memory; Create grade needs it in CPU memory.");
        if (img.row_bytes < 0) return fail("FIXER: unsupported image row layout.");
        int fw = img.bounds.x2 - img.bounds.x1, fh = img.bounds.y2 - img.bounds.y1;
        if (fw <= 0 || fh <= 0 || img.row_bytes < fw * 16 || img.row_bytes % 4)
            return fail("FIXER: cannot read the current frame for the reference match.");
        ms[0] = now_ms() - t0;
        t0 = now_ms();
        std::vector<double> down = fxm::downsample(reinterpret_cast<const float*>(img.data), img.row_bytes / 4, fw, fh, 4, 4, 1.0, dw, dh);
        host.assign(down.begin(), down.end());
        ms[1] = now_ms() - t0;
    }
    prog.step("Analysing frame", 20);
    t0 = now_ms();
    bool only = get_choice(effect, "matchSource", t) == 1;
    {
        DeviceScope scope;
        int current = 0, device = -1;
        cudaGetDevice(&current);
        {
            std::lock_guard<std::mutex> guard(inst->lock);
            device = inst->device;
        }
        if (device >= 0 && device != current) {
            if (cudaSetDevice(device) != cudaSuccess) return fail("FIXER: cannot select the GPU used for rendering.");
            scope.prev = current;
        }
        fe::LutSet luts;
        cudaError_t e = fill_film(film_key(effect, profile, t), luts);
        if (e != cudaSuccess) return fail(lut_error(e));
        fe::RenderParams p = base_params(effect, t);
        p.out_space = 3;
        p.impact = 1.0f;
        if (only) p.reserved2[0] = fe::LutSet::kFilmBypass;
        size_t pitch = static_cast<size_t>(dw) * 16, bytes = pitch * dh;
        float *d_src = nullptr, *d_dst = nullptr;
        e = cudaMalloc(&d_src, bytes);
        if (e == cudaSuccess) e = cudaMalloc(&d_dst, bytes);
        if (e == cudaSuccess) e = cudaMemcpy(d_src, host.data(), bytes, cudaMemcpyHostToDevice);
        if (e == cudaSuccess) e = fe::render(d_src, pitch, d_dst, pitch, dw, dh, p, luts, nullptr);
        if (e == cudaSuccess) e = cudaMemcpy(host.data(), d_dst, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_src);
        cudaFree(d_dst);
        if (e != cudaSuccess) return fail(cudaGetErrorString(e));
    }
    ms[2] = now_ms() - t0;
    prog.step("Reading reference", 45);
    t0 = now_ms();
    std::vector<uint8_t> ref;
    int rw = 0, rh = 0;
    int rs_status = read_reference(widen(ref_path), ref, rw, rh);
    if (rs_status == 2) return fail("FIXER: the reference image is too large (more than 30000 pixels on a side).");
    if (rs_status != 0) return fail("FIXER: cannot read the reference image. Use a JPEG or PNG file exported as sRGB.");
    ms[3] = now_ms() - t0;
    prog.step("Fitting look", 65);
    t0 = now_ms();
    std::vector<double> graded(host.begin(), host.end());
    fxm::Stats fs = fxm::stats(graded.data(), static_cast<size_t>(dw) * dh, 4, false);
    int qw = 0, qh = 0;
    std::vector<double> ref_small = fxm::downsample(ref.data(), static_cast<std::ptrdiff_t>(rw) * 4, rw, rh, 4, 3, 255.0, qw, qh);
    fxm::Stats rs = fxm::stats(ref_small.data(), static_cast<size_t>(qw) * qh, 3, true);
    fxm::State fitted = fxm::fit(fs, rs, only ? "only" : "film");
    fitted = fxm::refine(fitted, graded.data(), static_cast<size_t>(dw) * dh, 4,
                         qw == dw && qh == dh ? ref_small.data() : nullptr);
    std::string json = fxm::to_json(fitted);
    ms[4] = now_ms() - t0;
    prog.step("Building LUT", 85);
    t0 = now_ms();
    double s = fxm::clip(get_double(effect, "matchStrength", t), 0.0, 1.0);
    double pr = fxm::clip(get_double(effect, "matchProtectSkin", t), 0.0, 1.0);
    {
        std::lock_guard<std::mutex> guard(inst->lock);
        cache_match(inst, json, s, pr, effect);
    }
    ms[5] = now_ms() - t0;
    t0 = now_ms();
    {
        EditBlock edit(effect, "Create grade");
        OfxParamHandle h = param_handle(effect, "matchState");
        if (!h || gParam->paramSetValue(h, json.c_str()) != kOfxStatOK) return fail("FIXER: cannot store the reference match.");
    }
    ms[6] = now_ms() - t0;
    prog.step("Done", 100);
    prog.close();
    char msg[400];
    std::snprintf(msg, sizeof msg,
                  "FIXER timing: create grade read frame %.1f ms, downsample %.1f ms, analysis render %.1f ms, reference decode %.1f ms, "
                  "fit %.1f ms, LUT bake %.1f ms, store %.1f ms, total %.1f ms (with progress window %.1f ms)",
                  ms[0], ms[1], ms[2], ms[3], ms[4], ms[5], ms[6], ms[0] + ms[1] + ms[2] + ms[3] + ms[4] + ms[5] + ms[6],
                  now_ms() - prog.t0);
    log_timing(effect, msg);
    if (get_choice(effect, "matchMode", t) == 1 && gMessage)
        gMessage->message(effect, kOfxMessageMessage, nullptr, "%s", "FIXER: Detailed mode is not available yet; the grade was made in Simple mode.");
    return kOfxStatOK;
}

OfxStatus save_look(OfxImageEffectHandle effect, OfxTime t) {
    std::string name = get_string(effect, "lookName", t);
    if (name.empty() || name.find_first_of("\\/:*?\"<>|") != std::string::npos || name.find_first_not_of(". ") == std::string::npos)
        return fail_msg(effect, "FIXER: type a look name first (without \\ / : * ? \" < > |).");
    std::string text = get_string(effect, "matchState", t);
    fxm::State st;
    std::string why;
    if (text.empty()) return fail_msg(effect, "FIXER: create a grade first, then save the look.");
    if (!fxm::parse_state(text, st, &why))
        return fail_msg(effect, ("FIXER: the stored reference match is invalid (" + why + "). Create a grade first, then save the look.").c_str());
    std::wstring dir = looks_dir(), wname = widen(name);
    if (dir.empty()) return fail_msg(effect, "FIXER: the APPDATA folder is not set.");
    if (wname.empty()) return fail_msg(effect, "FIXER: the look name cannot be read.");
    Progress prog(effect, "FIXER: Save look");
    prog.step("Writing look files", 0);
    double t0 = now_ms();
    CreateDirectoryW(dir.substr(0, dir.find_last_of(L'\\')).c_str(), nullptr);
    CreateDirectoryW(dir.c_str(), nullptr);
    std::wstring base = dir + L"\\" + wname;
    double s = fxm::clip(get_double(effect, "matchStrength", t), 0.0, 1.0);
    double pr = fxm::clip(get_double(effect, "matchProtectSkin", t), 0.0, 1.0);
    if (!write_file(base + L".json", text) || !write_file(base + L".cube", fxm::cube_text(st, s, pr))) {
        prog.close();
        return fail_msg(effect, "FIXER: cannot write the look files in %APPDATA%\\FIXER\\looks.");
    }
    double took = now_ms() - t0;
    prog.step("Done", 100);
    prog.close();
    char msg[96];
    std::snprintf(msg, sizeof msg, "FIXER timing: save look %.1f ms", took);
    log_timing(effect, msg);
    return kOfxStatOK;
}

OfxStatus load_look_file(OfxImageEffectHandle effect, const std::wstring& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();
    fxm::State st;
    std::string why = "the file cannot be opened";
    if (!f || !fxm::parse_state(text, st, &why))
        return fail_msg(effect, ("FIXER: cannot read the look (" + why + "). Pick a .json file saved by FIXER.").c_str());
    EditBlock edit(effect, "Load look");
    OfxParamHandle h = param_handle(effect, "matchState"), src = param_handle(effect, "matchSource");
    if (!h || gParam->paramSetValue(h, text.c_str()) != kOfxStatOK) return fail_msg(effect, "FIXER: cannot store the reference match.");
    if (src) gParam->paramSetValue(src, st.source == "only" ? 1 : 0);
    sync_choice_buttons(effect, 0.0);
    return kOfxStatOK;
}

OfxStatus load_look(OfxImageEffectHandle effect, OfxTime t) {
    std::string path = get_string(effect, "lookLoad", t);
    if (path.empty()) return kOfxStatOK;
    return load_look_file(effect, widen(path));
}

OfxStatus instance_changed(OfxImageEffectHandle effect, OfxPropertySetHandle in) {
    char* type = nullptr;
    char* name = nullptr;
    char* reason = nullptr;
    gProp->propGetString(in, kOfxPropType, 0, &type);
    gProp->propGetString(in, kOfxPropName, 0, &name);
    gProp->propGetString(in, kOfxPropChangeReason, 0, &reason);
    if (type && name && !std::strcmp(type, kOfxTypeParameter)) {
        OfxTime t0 = 0;
        gProp->propGetDouble(in, kOfxPropTime, 0, &t0);
        bool user = reason && std::strcmp(reason, kOfxChangePluginEdited);
        for (const char* choice_name : kButtonChoices) {
            if (gButtonRows && !std::strcmp(name, choice_row_name(choice_name).c_str())) {
                if (user) {
                    const std::string row = get_string(effect, name, t0);
                    if (row != "0" && row != "1") return kOfxStatErrValue;
                    int selected = row == "1" ? 1 : 0;
                    if (get_choice(effect, choice_name, t0) != selected) {
                        EditBlock edit(effect, "Select mode");
                        if (gParam->paramSetValue(param_handle(effect, choice_name), selected) != kOfxStatOK)
                            return kOfxStatFailed;
                    }
                    sync_choice_buttons(effect, t0);
                }
                return kOfxStatOK;
            }
            if (!std::strcmp(name, choice_name)) {
                // Includes undo/redo, restored values and animated time changes.
                sync_choice_buttons(effect, t0);
                return kOfxStatOK;
            }
            for (int i = 0; i < 2; ++i) {
                if (!std::strcmp(name, choice_button_name(choice_name, i).c_str()) && user) {
                    if (get_choice(effect, choice_name, t0) != i) {
                        EditBlock edit(effect, "Select mode");
                        OfxParamHandle choice = param_handle(effect, choice_name);
                        if (!choice || gParam->paramSetValue(choice, i) != kOfxStatOK) return kOfxStatFailed;
                    }
                    sync_choice_buttons(effect, t0);
                    return kOfxStatOK;
                }
            }
        }

        if (!std::strcmp(name, "matchCreate")) return create_grade(effect, t0);
        if (!std::strcmp(name, "lookSave")) return save_look(effect, t0);
        if (!std::strcmp(name, "lookLoad") && user) return load_look(effect, t0);
        if (!std::strcmp(name, "lookSaved") && user) {
            int i = get_choice(effect, "lookSaved", t0);
            if (i > 0 && i <= static_cast<int>(gSavedLooks.size())) return load_look_file(effect, gSavedLooks[i - 1]);
            return kOfxStatOK;
        }
    }
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
    gProgress = static_cast<OfxProgressSuiteV1*>(const_cast<void*>(gHost->fetchSuite(gHost->host, kOfxProgressSuite, 1)));
    if (!gProp || !gEffect || !gParam) return kOfxStatErrMissingHostFeature;
    int row_animation = 0;
    gButtonRows = gProp->propGetInt(gHost->host, kButtonRowSupport, 0, &row_animation) == kOfxStatOK;
    load_profiles();
    return kOfxStatOK;
}

OfxStatus dispatch(const char* action, OfxImageEffectHandle effect, OfxPropertySetHandle in, OfxPropertySetHandle out) {
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

OfxStatus main_entry(const char* action, const void* handle, OfxPropertySetHandle in, OfxPropertySetHandle out) {
    auto effect = static_cast<OfxImageEffectHandle>(const_cast<void*>(handle));
    if (!action) return kOfxStatErrBadHandle;
    try {
        return dispatch(action, effect, in, out);
    } catch (...) {
        if (effect && gEffect && std::strcmp(action, kOfxActionLoad) && std::strcmp(action, kOfxActionUnload))
            post_error(effect, "FIXER: internal error; the action was stopped.");
        return kOfxStatFailed;
    }
}

void set_host(OfxHost* host) { gHost = host; }

OfxPlugin gPlugin = {kOfxImageEffectPluginApi, 1, kPluginId, 0, 1, set_host, main_entry};

}

OfxExport int OfxGetNumberOfPlugins(void) { return 1; }

OfxExport OfxPlugin* OfxGetPlugin(int nth) { return nth == 0 ? &gPlugin : nullptr; }
