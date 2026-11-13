#define NOMINMAX
#include <windows.h>

#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ofxCore.h"
#include "ofxGPURender.h"
#include "ofxImageEffect.h"
#include "ofxMessage.h"
#include "ofxParam.h"
#include "ofxProgress.h"
#include "ofxProperty.h"

namespace {

struct Value {
    std::vector<std::string> s;
    std::vector<double> d;
    std::vector<int> i;
    std::vector<void*> p;
};

struct PropSet {
    std::map<std::string, Value> v;
};

struct Param {
    std::string type;
    PropSet props;
    double d = 0.0;
    int i = 0;
    std::string s;
};

struct Clip {
    PropSet props;
    PropSet image;
    bool fetched = false;
};

struct Effect {
    PropSet props;
    std::map<std::string, std::unique_ptr<Param>> params;
    std::map<std::string, std::unique_ptr<Clip>> clips;
};

std::vector<std::string> gMessages;

PropSet* ps(OfxPropertySetHandle h) { return reinterpret_cast<PropSet*>(h); }

template <class T>
OfxStatus set_at(std::vector<T>& v, int idx, T x) {
    if (idx < 0) return kOfxStatErrBadIndex;
    if (static_cast<size_t>(idx) >= v.size()) v.resize(idx + 1);
    v[idx] = x;
    return kOfxStatOK;
}

template <class T>
OfxStatus get_at(const std::vector<T>& v, int idx, T* x) {
    if (idx < 0 || static_cast<size_t>(idx) >= v.size()) return kOfxStatErrBadIndex;
    *x = v[idx];
    return kOfxStatOK;
}

Value* find(OfxPropertySetHandle h, const char* n) {
    auto it = ps(h)->v.find(n);
    return it == ps(h)->v.end() ? nullptr : &it->second;
}

OfxStatus setPointer(OfxPropertySetHandle h, const char* n, int i, void* x) { return set_at(ps(h)->v[n].p, i, x); }
OfxStatus setString(OfxPropertySetHandle h, const char* n, int i, const char* x) { return set_at(ps(h)->v[n].s, i, std::string(x)); }
OfxStatus setDouble(OfxPropertySetHandle h, const char* n, int i, double x) { return set_at(ps(h)->v[n].d, i, x); }
OfxStatus setInt(OfxPropertySetHandle h, const char* n, int i, int x) { return set_at(ps(h)->v[n].i, i, x); }
OfxStatus getPointer(OfxPropertySetHandle h, const char* n, int i, void** x) {
    Value* v = find(h, n);
    return v ? get_at(v->p, i, x) : kOfxStatErrUnknown;
}
OfxStatus getString(OfxPropertySetHandle h, const char* n, int i, char** x) {
    Value* v = find(h, n);
    if (!v || i < 0 || static_cast<size_t>(i) >= v->s.size()) return kOfxStatErrUnknown;
    *x = const_cast<char*>(v->s[i].c_str());
    return kOfxStatOK;
}
OfxStatus getDouble(OfxPropertySetHandle h, const char* n, int i, double* x) {
    Value* v = find(h, n);
    return v ? get_at(v->d, i, x) : kOfxStatErrUnknown;
}
OfxStatus getInt(OfxPropertySetHandle h, const char* n, int i, int* x) {
    Value* v = find(h, n);
    return v ? get_at(v->i, i, x) : kOfxStatErrUnknown;
}
OfxStatus getIntN(OfxPropertySetHandle h, const char* n, int c, int* x) {
    for (int k = 0; k < c; ++k)
        if (OfxStatus s = getInt(h, n, k, x + k); s != kOfxStatOK) return s;
    return kOfxStatOK;
}
OfxStatus getDoubleN(OfxPropertySetHandle h, const char* n, int c, double* x) {
    for (int k = 0; k < c; ++k)
        if (OfxStatus s = getDouble(h, n, k, x + k); s != kOfxStatOK) return s;
    return kOfxStatOK;
}
OfxStatus unsupported() { return kOfxStatErrUnsupported; }

OfxPropertySuiteV1 gProp = {
    setPointer, setString, setDouble, setInt,
    [](OfxPropertySetHandle, const char*, int, void* const*) { return unsupported(); },
    [](OfxPropertySetHandle, const char*, int, const char* const*) { return unsupported(); },
    [](OfxPropertySetHandle, const char*, int, const double*) { return unsupported(); },
    [](OfxPropertySetHandle, const char*, int, const int*) { return unsupported(); },
    getPointer, getString, getDouble, getInt,
    [](OfxPropertySetHandle, const char*, int, void**) { return unsupported(); },
    [](OfxPropertySetHandle, const char*, int, char**) { return unsupported(); },
    getDoubleN, getIntN,
    [](OfxPropertySetHandle, const char*) { return unsupported(); },
    [](OfxPropertySetHandle h, const char* n, int* c) {
        Value* v = find(h, n);
        if (!v) return kOfxStatErrUnknown;
        *c = static_cast<int>(v->s.size() + v->d.size() + v->i.size() + v->p.size());
        return kOfxStatOK;
    }};

Effect* fx(OfxImageEffectHandle h) { return reinterpret_cast<Effect*>(h); }

OfxStatus fx_get_props(OfxImageEffectHandle hd, OfxPropertySetHandle* p) {
    *p = reinterpret_cast<OfxPropertySetHandle>(&fx(hd)->props);
    return kOfxStatOK;
}
OfxStatus fx_get_params(OfxImageEffectHandle hd, OfxParamSetHandle* p) {
    *p = reinterpret_cast<OfxParamSetHandle>(hd);
    return kOfxStatOK;
}
OfxStatus fx_clip_define(OfxImageEffectHandle hd, const char* name, OfxPropertySetHandle* p) {
    auto& c = fx(hd)->clips[name];
    c = std::make_unique<Clip>();
    *p = reinterpret_cast<OfxPropertySetHandle>(&c->props);
    return kOfxStatOK;
}
OfxStatus fx_clip_handle(OfxImageEffectHandle hd, const char* name, OfxImageClipHandle* c, OfxPropertySetHandle* p) {
    auto it = fx(hd)->clips.find(name);
    if (it == fx(hd)->clips.end()) return kOfxStatErrUnknown;
    *c = reinterpret_cast<OfxImageClipHandle>(it->second.get());
    if (p) *p = reinterpret_cast<OfxPropertySetHandle>(&it->second->props);
    return kOfxStatOK;
}
OfxStatus fx_clip_image(OfxImageClipHandle c, OfxTime, const OfxRectD*, OfxPropertySetHandle* img) {
    Clip* clip = reinterpret_cast<Clip*>(c);
    if (clip->image.v.empty()) return kOfxStatFailed;
    clip->fetched = true;
    *img = reinterpret_cast<OfxPropertySetHandle>(&clip->image);
    return kOfxStatOK;
}
OfxStatus fx_clip_release(OfxPropertySetHandle) { return kOfxStatOK; }
OfxStatus fx_clip_rod(OfxImageClipHandle c, OfxTime, OfxRectD* rod) {
    Clip* clip = reinterpret_cast<Clip*>(c);
    int b[4] = {0, 0, 0, 0};
    if (getIntN(reinterpret_cast<OfxPropertySetHandle>(&clip->image), kOfxImagePropBounds, 4, b) != kOfxStatOK) return kOfxStatFailed;
    *rod = {double(b[0]), double(b[1]), double(b[2]), double(b[3])};
    return kOfxStatOK;
}

OfxImageEffectSuiteV1 make_effect_suite() {
    OfxImageEffectSuiteV1 e{};
    e.getPropertySet = fx_get_props;
    e.getParamSet = fx_get_params;
    e.clipDefine = fx_clip_define;
    e.clipGetHandle = fx_clip_handle;
    e.clipGetImage = fx_clip_image;
    e.clipReleaseImage = fx_clip_release;
    e.clipGetRegionOfDefinition = fx_clip_rod;
    return e;
}

OfxImageEffectSuiteV1 gEffect = make_effect_suite();

Param* param_of(OfxParamHandle h) { return reinterpret_cast<Param*>(h); }

OfxStatus paramGetValueAtTime(OfxParamHandle h, OfxTime time, ...) {
    va_list ap;
    va_start(ap, time);
    Param* p = param_of(h);
    if (p->type == kOfxParamTypeDouble) *va_arg(ap, double*) = p->d;
    else if ((p->type == kOfxParamTypeString || p->type == "OfxParamTypeButtonRow")) *va_arg(ap, char**) = p->s.data();
    else *va_arg(ap, int*) = p->i;
    va_end(ap);
    return kOfxStatOK;
}

OfxStatus paramSetValue(OfxParamHandle h, ...) {
    va_list ap;
    va_start(ap, h);
    Param* p = param_of(h);
    if (p->type == kOfxParamTypeDouble) p->d = va_arg(ap, double);
    else if ((p->type == kOfxParamTypeString || p->type == "OfxParamTypeButtonRow")) p->s = va_arg(ap, const char*);
    else p->i = va_arg(ap, int);
    va_end(ap);
    return kOfxStatOK;
}

OfxStatus param_define(OfxParamSetHandle st, const char* type, const char* name, OfxPropertySetHandle* props) {
    auto& p = reinterpret_cast<Effect*>(st)->params[name];
    p = std::make_unique<Param>();
    p->type = type;
    if (props) *props = reinterpret_cast<OfxPropertySetHandle>(&p->props);
    return kOfxStatOK;
}
OfxStatus param_handle(OfxParamSetHandle st, const char* name, OfxParamHandle* hd, OfxPropertySetHandle* props) {
    auto& m = reinterpret_cast<Effect*>(st)->params;
    auto it = m.find(name);
    if (it == m.end()) return kOfxStatErrUnknown;
    *hd = reinterpret_cast<OfxParamHandle>(it->second.get());
    if (props) *props = reinterpret_cast<OfxPropertySetHandle>(&it->second->props);
    return kOfxStatOK;
}

int gEditBegin = 0, gEditEnd = 0;
std::string gEditLabels;

OfxStatus edit_begin(OfxParamSetHandle, const char* name) {
    ++gEditBegin;
    gEditLabels += std::string(name ? name : "") + ";";
    return kOfxStatOK;
}
OfxStatus edit_end(OfxParamSetHandle) {
    ++gEditEnd;
    return kOfxStatOK;
}

OfxParameterSuiteV1 make_param_suite() {
    OfxParameterSuiteV1 s{};
    s.paramEditBegin = edit_begin;
    s.paramEditEnd = edit_end;
    s.paramDefine = param_define;
    s.paramGetHandle = param_handle;
    s.paramGetValueAtTime = paramGetValueAtTime;
    s.paramSetValue = paramSetValue;
    return s;
}

OfxParameterSuiteV1 gParam = make_param_suite();

OfxStatus message(void*, const char* type, const char*, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    gMessages.push_back(std::string(type) + ": " + buf);
    return kOfxStatOK;
}

OfxMessageSuiteV1 gMessage = {message};

bool gOfferProgress = false;
int gProgStarts = 0, gProgUpdates = 0, gProgEnds = 0;
OfxStatus prog_start(void*, const char*) {
    ++gProgStarts;
    return kOfxStatOK;
}
OfxStatus prog_update(void*, double) {
    ++gProgUpdates;
    return kOfxStatOK;
}
OfxStatus prog_end(void*) {
    ++gProgEnds;
    return kOfxStatOK;
}
OfxProgressSuiteV1 gProgress = {prog_start, prog_update, prog_end};

HHOOK gHook = nullptr;
int gWindows = 0;
ULONGLONG gWinStart = 0, gWinMs = 0;

bool is_progress_window(HWND w) {
    wchar_t cls[64] = {};
    return GetClassNameW(w, cls, 64) && !std::wcscmp(cls, L"FIXERProgress");
}

LRESULT CALLBACK cbt(int code, WPARAM w, LPARAM l) {
    if (code == HCBT_CREATEWND && is_progress_window(reinterpret_cast<HWND>(w))) {
        ++gWindows;
        gWinStart = GetTickCount64();
    }
    if (code == HCBT_DESTROYWND && is_progress_window(reinterpret_cast<HWND>(w))) gWinMs += GetTickCount64() - gWinStart;
    return CallNextHookEx(gHook, code, w, l);
}

const void* fetchSuite(OfxPropertySetHandle, const char* name, int version) {
    if (version != 1) return nullptr;
    if (!std::strcmp(name, kOfxPropertySuite)) return &gProp;
    if (!std::strcmp(name, kOfxImageEffectSuite)) return &gEffect;
    if (!std::strcmp(name, kOfxParameterSuite)) return &gParam;
    if (!std::strcmp(name, kOfxMessageSuite)) return &gMessage;
    if (gOfferProgress && !std::strcmp(name, kOfxProgressSuite)) return &gProgress;
    return nullptr;
}

PropSet gHostProps;
OfxHost gHost = {reinterpret_cast<OfxPropertySetHandle>(&gHostProps), fetchSuite};

bool read_file(const char* path, std::vector<char>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    out.resize(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    return static_cast<bool>(f.read(out.data(), out.size()));
}

OfxPropertySetHandle h(PropSet& p) { return reinterpret_cast<OfxPropertySetHandle>(&p); }

void set_image(Clip& c, void* data, int row_bytes, int w, int hgt, const char* premult) {
    c.image = PropSet();
    setPointer(h(c.image), kOfxImagePropData, 0, data);
    setInt(h(c.image), kOfxImagePropRowBytes, 0, row_bytes);
    int b[4] = {0, 0, w, hgt};
    for (int k = 0; k < 4; ++k) setInt(h(c.image), kOfxImagePropBounds, k, b[k]);
    setString(h(c.image), kOfxImageEffectPropPixelDepth, 0, kOfxBitDepthFloat);
    setString(h(c.image), kOfxImageEffectPropComponents, 0, kOfxImageComponentRGBA);
    setString(h(c.image), kOfxImageEffectPropPreMultiplication, 0, premult);
}

}

int main(int argc, char** argv) {
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    std::vector<std::string> args8;
    std::vector<char*> argv8;
    if (wargv && wargc == argc) {
        for (int k = 0; k < wargc; ++k) {
            int n = WideCharToMultiByte(CP_UTF8, 0, wargv[k], -1, nullptr, 0, nullptr, nullptr);
            std::string a(n > 0 ? static_cast<size_t>(n) : 1, '\0');
            if (n > 0) WideCharToMultiByte(CP_UTF8, 0, wargv[k], -1, a.data(), n, nullptr, nullptr);
            a.resize(std::strlen(a.c_str()));
            args8.push_back(a);
        }
        for (auto& a : args8) argv8.push_back(a.data());
        argv8.push_back(nullptr);
        argv = argv8.data();
    }
    if (wargv) LocalFree(wargv);
    const char* profile_id = nullptr;
    const char *look_pick = nullptr, *create_device = nullptr, *press_button = nullptr;
    bool check_buttons = false, check_rows = false;
    int light = 0, balance = 0;
    const char *match_state = nullptr, *match_strength = nullptr, *match_protect = nullptr, *match_source = nullptr,
               *match_show = nullptr, *match_create = nullptr, *look_save = nullptr, *look_load = nullptr;
    while (argc > 3 && argv[argc - 2][0] == '-' && argv[argc - 2][1] == '-') {
        if (!std::strcmp(argv[argc - 2], "--profile-id")) profile_id = argv[argc - 1];
        else if (!std::strcmp(argv[argc - 2], "--light")) light = std::atoi(argv[argc - 1]);
        else if (!std::strcmp(argv[argc - 2], "--balance")) balance = std::atoi(argv[argc - 1]);
        else if (!std::strcmp(argv[argc - 2], "--match-state")) match_state = argv[argc - 1];
        else if (!std::strcmp(argv[argc - 2], "--match-strength")) match_strength = argv[argc - 1];
        else if (!std::strcmp(argv[argc - 2], "--match-protect")) match_protect = argv[argc - 1];
        else if (!std::strcmp(argv[argc - 2], "--match-source")) match_source = argv[argc - 1];
        else if (!std::strcmp(argv[argc - 2], "--match-show")) match_show = argv[argc - 1];
        else if (!std::strcmp(argv[argc - 2], "--match-create")) match_create = argv[argc - 1];
        else if (!std::strcmp(argv[argc - 2], "--look-save")) look_save = argv[argc - 1];
        else if (!std::strcmp(argv[argc - 2], "--look-load")) look_load = argv[argc - 1];
        else if (!std::strcmp(argv[argc - 2], "--look-pick")) look_pick = argv[argc - 1];
        else if (!std::strcmp(argv[argc - 2], "--match-create-device")) create_device = argv[argc - 1];
        else if (!std::strcmp(argv[argc - 2], "--press")) press_button = argv[argc - 1];
        else if (!std::strcmp(argv[argc - 2], "--ui-rows")) check_rows = std::atoi(argv[argc - 1]) != 0;
        else if (!std::strcmp(argv[argc - 2], "--ui-buttons")) check_buttons = std::atoi(argv[argc - 1]) != 0;
        else if (!std::strcmp(argv[argc - 2], "--progress-suite")) gOfferProgress = std::atoi(argv[argc - 1]) != 0;
        else break;
        argc -= 2;
    }
    if (argc != 17 && argc != 21 && argc != 24) {
        std::fprintf(stderr,
                     "usage: ofx_test_host plugin.ofx in.raw out.raw width height in_space out_space exposure push_pull impact premult "
                     "win_x1 win_y1 win_x2 win_y2 profile_index\n");
        return 1;
    }
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) return std::fprintf(stderr, "LoadLibrary failed %lu\n", GetLastError()), 2;
    auto count = reinterpret_cast<int (*)()>(GetProcAddress(dll, "OfxGetNumberOfPlugins"));
    auto get = reinterpret_cast<OfxPlugin* (*)(int)>(GetProcAddress(dll, "OfxGetPlugin"));
    if (!count || !get || count() != 1) return std::fprintf(stderr, "bad exports\n"), 2;
    OfxPlugin* plug = get(0);
    std::printf("plugin=%s api=%s v%u.%u\n", plug->pluginIdentifier, plug->pluginApi, plug->pluginVersionMajor,
                plug->pluginVersionMinor);
    if (check_rows) setInt(h(gHostProps), "OfxParamHostPropSupportsButtonRowAnimation", 0, 0);
    plug->setHost(&gHost);
    Effect desc, inst;
    auto call = [&](const char* action, Effect& e, PropSet* in, PropSet* out) {
        return plug->mainEntry(action, &e, in ? h(*in) : nullptr, out ? h(*out) : nullptr);
    };
    if (call(kOfxActionLoad, desc, nullptr, nullptr) != kOfxStatOK) return std::fprintf(stderr, "load failed\n"), 3;
    if (call(kOfxActionDescribe, desc, nullptr, nullptr) != kOfxStatOK) return std::fprintf(stderr, "describe failed\n"), 3;
    char* cuda = nullptr;
    getString(h(desc.props), kOfxImageEffectPropCudaRenderSupported, 0, &cuda);
    std::printf("cuda_render_supported=%s\n", cuda ? cuda : "(none)");
    if (call(kOfxImageEffectActionDescribeInContext, inst, nullptr, nullptr) != kOfxStatOK) return std::fprintf(stderr, "context failed\n"), 3;
    std::printf("params=%zu clips=%zu\n", inst.params.size(), inst.clips.size());
    if (Value* opts = find(h(inst.params["profile"]->props), kOfxParamPropChoiceOption))
        for (size_t k = 0; k < opts->s.size(); ++k) std::printf("profile_option %zu %s\n", k, opts->s[k].c_str());
    if (Value* opts = find(h(inst.params["lookSaved"]->props), kOfxParamPropChoiceOption))
        for (size_t k = 0; k < opts->s.size(); ++k) std::printf("look_option %zu %s\n", k, opts->s[k].c_str());
    inst.params["profile"]->i = std::atoi(argv[16]);
    if (profile_id) inst.params["profileId"]->s = profile_id;
    if (call(kOfxActionCreateInstance, inst, nullptr, nullptr) != kOfxStatOK) return std::fprintf(stderr, "create failed\n"), 3;
    if (profile_id) {
        std::printf("profile_choice_after_create=%d\n", inst.params["profile"]->i);
    } else {
        inst.params["profile"]->i = std::atoi(argv[16]);
        PropSet change;
        setString(h(change), kOfxPropType, 0, kOfxTypeParameter);
        setString(h(change), kOfxPropName, 0, "profile");
        setString(h(change), kOfxPropChangeReason, 0, kOfxChangeUserEdited);
        setDouble(h(change), kOfxPropTime, 0, 0.0);
        call(kOfxActionInstanceChanged, inst, &change, nullptr);
        std::printf("profile_id=%s\n", inst.params["profileId"]->s.c_str());
    }
    inst.params["inSpace"]->i = std::atoi(argv[6]);
    inst.params["outSpace"]->i = std::atoi(argv[7]);
    inst.params["exposure"]->d = std::atof(argv[8]);
    inst.params["pushPull"]->d = std::atof(argv[9]);
    inst.params["impact"]->d = std::atof(argv[10]);
    double time = 0.0;
    inst.params["grainEnable"]->i = 0;
    inst.params["shootingLight"]->i = light;
    inst.params["greyBalance"]->i = balance;
    inst.params["labScan"]->i = 0;
    if (argc == 24) {
        inst.params["halationAmount"]->d = std::atof(argv[21]);
        inst.params["halationRadius"]->d = std::atof(argv[22]);
        inst.params["bloomAmount"]->d = std::atof(argv[23]);
    }
    if (argc >= 21) {
        inst.params["grainAmount"]->d = std::atof(argv[17]);
        inst.params["grainEnable"]->i = inst.params["grainAmount"]->d > 0.0;
        inst.params["grainSize"]->d = std::atof(argv[18]);
        inst.params["grainSeed"]->i = std::atoi(argv[19]);
        time = std::atof(argv[20]);
    }
    PropSet in_args, out_args;
    setDouble(h(in_args), kOfxPropTime, 0, 0.0);
    int ident = call(kOfxImageEffectActionIsIdentity, inst, &in_args, &out_args);
    std::printf("is_identity=%s\n", ident == kOfxStatOK ? "yes" : "no");
    int width = std::atoi(argv[4]), height = std::atoi(argv[5]);
    std::vector<char> img, out_img;
    size_t row = static_cast<size_t>(width) * 16;
    if (!read_file(argv[2], img) || img.size() != row * height) return std::fprintf(stderr, "bad image\n"), 1;
    out_img.assign(img.size(), 0);
    for (size_t k = 0; k < out_img.size() / 4; ++k) reinterpret_cast<float*>(out_img.data())[k] = -7.0f;
    float *src = nullptr, *dst = nullptr;
    size_t sp = 0, dp = 0;
    cudaMallocPitch(&src, &sp, row, height);
    cudaMallocPitch(&dst, &dp, row, height);
    cudaMemcpy2D(src, sp, img.data(), row, row, height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dst, dp, out_img.data(), row, row, height, cudaMemcpyHostToDevice);
    const char* premult = std::atoi(argv[11]) ? kOfxImagePreMultiplied : kOfxImageUnPreMultiplied;
    auto param_changed = [&](const char* pname) {
        PropSet change;
        setString(h(change), kOfxPropType, 0, kOfxTypeParameter);
        setString(h(change), kOfxPropName, 0, pname);
        setString(h(change), kOfxPropChangeReason, 0, kOfxChangeUserEdited);
        setDouble(h(change), kOfxPropTime, 0, time);
        return call(kOfxActionInstanceChanged, inst, &change, nullptr);
    };
    if (check_rows) {
        bool valid = true;
        for (const char* name : {"glowQuality", "matchSource", "matchMode", "matchShow"}) {
            const bool glow = !std::strcmp(name, "glowQuality");
            const std::string row = std::string(name) + "Row";
            auto found = inst.params.find(row);
            if (found == inst.params.end()) return 4;
            Param& ui = *found->second;
            valid = valid && ui.type == "OfxParamTypeButtonRow";
            char* label = nullptr;
            getString(h(ui.props), kOfxPropLabel, 0, &label);
            valid = valid && label && (glow ? std::string(label) == "Glow:" : std::string(label).back() == ':');
            char* option = nullptr;
            getString(h(ui.props), kOfxParamPropChoiceOption, 0, &option);
            if (glow) valid = valid && option && std::string(option) == "Fast";
            const auto* enums = find(h(ui.props), "OfxParamPropChoiceEnum");
            const auto* icons = find(h(ui.props), "OfxParamPropChoiceIcon");
            valid = valid && enums && enums->s.size() == 2 && icons && icons->s.size() == 16;
            valid = valid && enums->s[0] == (glow ? "1" : "0") && enums->s[1] == (glow ? "0" : "1");
            int previous = inst.params[name]->i;
            valid = valid && ui.s == std::to_string(previous);
            for (int selected : {0, 1, 0}) {
                ui.s = enums->s[selected];
                valid = valid && param_changed(row.c_str()) == kOfxStatOK;
                valid = valid && inst.params[name]->i == (glow ? 1 - selected : selected);
            }
            // Saved original Choice values remain authoritative on restore/undo.
            inst.params[name]->i = previous;
            param_changed(name);
            valid = valid && ui.s == std::to_string(previous);
            for (int i = 0; i < 2; ++i) {
                int secret = 0;
                getInt(h(inst.params[std::string(name) + "Option" + std::to_string(i)]->props), kOfxParamPropSecret, 0, &secret);
                valid = valid && secret == 1;
            }
        }
        std::printf("mode_rows_valid=%d\n", valid ? 1 : 0);
        if (!valid) return 4;
    }
    if (check_buttons) {
        bool valid = true;
        for (const char* name : {"glowQuality", "matchSource", "matchMode", "matchShow"}) {
            int secret = 0, previous = inst.params[name]->i;
            getInt(h(inst.params[name]->props), kOfxParamPropSecret, 0, &secret);
            valid = valid && secret == 1 && inst.params[name]->type == kOfxParamTypeChoice;
            for (int selected : {1, 0, 1}) {
                std::string id = std::string(name) + "Option" + std::to_string(selected);
                auto button = inst.params.find(id);
                valid = valid && button != inst.params.end() && button->second->type == kOfxParamTypePushButton;
                valid = valid && param_changed(id.c_str()) == kOfxStatOK && inst.params[name]->i == selected;
                for (int i = 0; i < 2; ++i) {
                    std::string other = std::string(name) + "Option" + std::to_string(i);
                    char* label = nullptr;
                    getString(h(inst.params[other]->props), kOfxPropLabel, 0, &label);
                    bool marked = label && std::string(label).find("\xE2\x97\x8F ") == 0;
                    valid = valid && marked == (i == selected);
                }
            }
            // Simulate restoring a saved value / undo without clicking a button.
            inst.params[name]->i = previous;
            param_changed(name);
            char* label = nullptr;
            getString(h(inst.params[std::string(name) + "Option" + std::to_string(previous)]->props), kOfxPropLabel, 0, &label);
            valid = valid && label && std::string(label).find("\xE2\x97\x8F ") == 0;
        }
        for (const char* name : {"inSpace", "outSpace", "grainRate", "profile", "lookSaved"}) {
            int secret = 0;
            getInt(h(inst.params[name]->props), kOfxParamPropSecret, 0, &secret);
            valid = valid && !secret && inst.params[name]->type == kOfxParamTypeChoice;
        }
        std::printf("mode_buttons_valid=%d\n", valid ? 1 : 0);
        if (!valid) return 4;
    }
    if (press_button) {
        auto button = inst.params.find(press_button);
        if (button == inst.params.end() || button->second->type != kOfxParamTypePushButton || param_changed(press_button) != kOfxStatOK) return 4;
    }
    if (match_strength) inst.params["matchStrength"]->d = std::atof(match_strength);
    if (match_protect) inst.params["matchProtectSkin"]->d = std::atof(match_protect);
    if (match_source) inst.params["matchSource"]->i = std::atoi(match_source);
    if (match_show) inst.params["matchShow"]->i = std::atoi(match_show);
    if (match_state) {
        std::vector<char> text;
        if (!read_file(match_state, text)) return std::fprintf(stderr, "bad match state\n"), 1;
        inst.params["matchState"]->s.assign(text.begin(), text.end());
    }
    gHook = SetWindowsHookExW(WH_CBT, cbt, nullptr, GetCurrentThreadId());
    if (look_load) {
        inst.params["lookLoad"]->s = look_load;
        std::printf("look_load_status=%d\n", param_changed("lookLoad"));
    }
    if (look_pick) {
        inst.params["lookSaved"]->i = std::atoi(look_pick);
        std::printf("look_pick_status=%d\n", param_changed("lookSaved"));
    }
    if (match_create) {
        inst.params["matchFile"]->s = match_create;
        if (create_device && std::atoi(create_device))
            set_image(*inst.clips[kOfxImageEffectSimpleSourceClipName], src, static_cast<int>(sp), width, height, premult);
        else
            set_image(*inst.clips[kOfxImageEffectSimpleSourceClipName], img.data(), static_cast<int>(row), width, height, premult);
        std::printf("match_create_status=%d\n", param_changed("matchCreate"));
    }
    if (look_save) {
        inst.params["lookName"]->s = look_save;
        std::printf("look_save_status=%d\n", param_changed("lookSave"));
    }
    if (gHook) UnhookWindowsHookEx(gHook);
    if (gProgStarts)
        std::printf("progress_path=host_suite starts=%d updates=%d ends=%d\n", gProgStarts, gProgUpdates, gProgEnds);
    else if (gWindows)
        std::printf("progress_path=own_window windows=%d shown_ms=%llu\n", gWindows, gWinMs);
    else if (match_create || look_save)
        std::printf("progress_path=none\n");
    if (gEditBegin || gEditEnd) std::printf("param_edit begin=%d end=%d labels=%s\n", gEditBegin, gEditEnd, gEditLabels.c_str());
    if (match_state || look_load || match_create || look_save || look_pick)
        std::printf("match_source=%d\nmatch_state=%s\n", inst.params["matchSource"]->i, inst.params["matchState"]->s.c_str());
    set_image(*inst.clips[kOfxImageEffectSimpleSourceClipName], src, static_cast<int>(sp), width, height, premult);
    set_image(*inst.clips[kOfxImageEffectOutputClipName], dst, static_cast<int>(dp), width, height, premult);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    PropSet rargs;
    setDouble(h(rargs), kOfxPropTime, 0, time);
    setDouble(h(rargs), kOfxImageEffectPropRenderScale, 0, 1.0);
    setDouble(h(rargs), kOfxImageEffectPropRenderScale, 1, 1.0);
    for (int k = 0; k < 4; ++k) setInt(h(rargs), kOfxImageEffectPropRenderWindow, k, std::atoi(argv[12 + k]));
    setInt(h(rargs), kOfxImageEffectPropCudaEnabled, 0, 0);
    int cpu = call(kOfxImageEffectActionRender, inst, &rargs, nullptr);
    std::printf("cpu_render_status=%d messages=%zu\n", cpu, gMessages.size());
    setInt(h(rargs), kOfxImageEffectPropCudaEnabled, 0, 1);
    setPointer(h(rargs), kOfxImageEffectPropCudaStream, 0, stream);
    int st = call(kOfxImageEffectActionRender, inst, &rargs, nullptr);
    cudaStreamSynchronize(stream);
    std::printf("cuda_render_status=%d\n", st);
    for (auto& m : gMessages) std::printf("message %s\n", m.c_str());
    cudaMemcpy2D(out_img.data(), row, dst, dp, row, height, cudaMemcpyDeviceToHost);
    std::ofstream(argv[3], std::ios::binary).write(out_img.data(), out_img.size());
    call(kOfxActionDestroyInstance, inst, nullptr, nullptr);
    call(kOfxActionUnload, desc, nullptr, nullptr);
    cudaStreamDestroy(stream);
    cudaFree(src);
    cudaFree(dst);
    return st == kOfxStatOK ? 0 : 4;
}
