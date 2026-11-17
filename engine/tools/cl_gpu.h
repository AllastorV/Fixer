#pragma once
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

struct ClGpu {
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    ClGpu() {
        cl_uint n = 0;
        clGetPlatformIDs(0, nullptr, &n);
        std::vector<cl_platform_id> platforms(n);
        if (n) clGetPlatformIDs(n, platforms.data(), nullptr);
        const char* want = std::getenv("FE_CL_PLATFORM");
        int pick = want ? std::atoi(want) : -1;
        for (cl_uint i = 0; i < n && !device; ++i)
            if (pick < 0 || static_cast<int>(i) == pick) clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (!device) {
            std::fprintf(stderr, "no OpenCL GPU\n");
            std::exit(2);
        }
        cl_int e = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &e);
        if (e == CL_SUCCESS) queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &e);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "OpenCL context failed %d\n", static_cast<int>(e));
            std::exit(2);
        }
    }
    ~ClGpu() {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
    cl_mem buffer(size_t bytes, const void* init) {
        cl_int e = CL_SUCCESS;
        cl_mem m = clCreateBuffer(context, CL_MEM_READ_WRITE | (init ? CL_MEM_COPY_HOST_PTR : 0), bytes, const_cast<void*>(init), &e);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "OpenCL buffer failed %d\n", static_cast<int>(e));
            std::exit(2);
        }
        return m;
    }
};
