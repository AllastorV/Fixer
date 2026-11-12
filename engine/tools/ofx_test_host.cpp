#define NOMINMAX
#include <windows.h>

#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    else if (p->type == kOfxParamTypeString) *va_arg(ap, char**) = p->s.data();
    else *va_arg(ap, int*) = p->i;
    va_end(ap);
    return kOfxStatOK;
}

OfxStatus paramSetValue(OfxParamHandle h, ...) {
    va_list ap;
    va_start(ap, h);
    Param* p = param_of(h);
    if (p->type == kOfxParamTypeDouble) p->d = va_arg(ap, double);
    else if (p->type == kOfxParamTypeString) p->s = va_arg(ap, const char*);
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

OfxParameterSuiteV1 make_param_suite() {
    OfxParameterSuiteV1 s{};
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

const void* fetchSuite(OfxPropertySetHandle, const char* name, int version) {
    if (version != 1) return nullptr;
    if (!std::strcmp(name, kOfxPropertySuite)) return &gProp;
    if (!std::strcmp(name, kOfxImageEffectSuite)) return &gEffect;
    if (!std::strcmp(name, kOfxParameterSuite)) return &gParam;
    if (!std::strcmp(name, kOfxMessageSuite)) return &gMessage;
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
    const char* profile_id = nullptr;
    int light = 0, balance = 0;
    while (argc > 3 && argv[argc - 2][0] == '-' && argv[argc - 2][1] == '-') {
        if (!std::strcmp(argv[argc - 2], "--profile-id")) profile_id = argv[argc - 1];
        else if (!std::strcmp(argv[argc - 2], "--light")) light = std::atoi(argv[argc - 1]);
        else if (!std::strcmp(argv[argc - 2], "--balance")) balance = std::atoi(argv[argc - 1]);
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
    inst.params["profile"]->i = std::atoi(argv[16]);
    if (profile_id) inst.params["profileId"]->s = profile_id;
    if (call(kOfxActionCreateInstance, inst, nullptr, nullptr) != kOfxStatOK) return std::fprintf(stderr, "create failed\n"), 3;
    if (profile_id) {
        std::printf("profile_choice_after_create=%d\n", inst.params["profile"]->i);
    } else {
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
