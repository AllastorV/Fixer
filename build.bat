@echo off
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0"
if not defined OpenCL_ROOT if defined CUDA_PATH set "OpenCL_ROOT=%CUDA_PATH%"
"C:\Program Files\CMake\bin\cmake.exe" -S . -B build\release -G Ninja -DCMAKE_BUILD_TYPE=Release || exit /b 1
"C:\Program Files\CMake\bin\cmake.exe" --build build\release || exit /b 1
