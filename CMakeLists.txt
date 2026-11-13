cmake_minimum_required(VERSION 3.26)
project(film_emulation_ofx LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(CUDAToolkit REQUIRED)
set(GEN_DIR ${CMAKE_BINARY_DIR}/gen)
add_custom_command(
  OUTPUT ${GEN_DIR}/spaces_gen.h
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GEN_DIR}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/bake/gen_header.py ${GEN_DIR}/spaces_gen.h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bake
  DEPENDS ${CMAKE_SOURCE_DIR}/bake/gen_header.py ${CMAKE_SOURCE_DIR}/bake/filmemu/spaces.py)

add_library(fe_engine STATIC engine/src/render.cu ${GEN_DIR}/spaces_gen.h)
target_include_directories(fe_engine PUBLIC engine/include PRIVATE ${GEN_DIR})
target_link_libraries(fe_engine PUBLIC CUDA::cudart_static)
set_target_properties(fe_engine PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_RUNTIME_LIBRARY Static)

set(BUNDLE_DIR ${CMAKE_BINARY_DIR}/Fixer.ofx.bundle)
set(FIXER_PREBUILT_PROFILES "" CACHE PATH "Directory containing prebuilt film profiles")
if(FIXER_PREBUILT_PROFILES)
  file(GLOB PROFILE_MANIFESTS "${FIXER_PREBUILT_PROFILES}/*/profile.json")
  if(NOT PROFILE_MANIFESTS)
    message(FATAL_ERROR "No film profiles found in ${FIXER_PREBUILT_PROFILES}")
  endif()
  set(PROFILES_DIR "${FIXER_PREBUILT_PROFILES}")
else()
  file(GLOB STOCK_DATA CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/bake/data/stocks/*.json)
  set(LUT_DIR ${CMAKE_BINARY_DIR}/luts)
  add_custom_command(
    OUTPUT ${LUT_DIR}/profiles.stamp
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/bake/bake_luts.py ${LUT_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bake
    DEPENDS ${CMAKE_SOURCE_DIR}/bake/bake_luts.py ${CMAKE_SOURCE_DIR}/bake/filmemu/film.py ${CMAKE_SOURCE_DIR}/bake/filmemu/lut.py
            ${CMAKE_SOURCE_DIR}/bake/filmemu/spaces.py ${CMAKE_SOURCE_DIR}/bake/filmemu/spectral.py
            ${CMAKE_SOURCE_DIR}/bake/filmemu/stocks.py ${CMAKE_SOURCE_DIR}/bake/filmemu/grain.py ${CMAKE_SOURCE_DIR}/bake/filmemu/bw.py ${CMAKE_SOURCE_DIR}/bake/filmemu/reversal.py ${STOCK_DATA}
            ${CMAKE_SOURCE_DIR}/bake/data/jakob_nodes_65.npz)
  set(PROFILES_DIR ${LUT_DIR}/profiles)
  set(PROFILES_STAMP ${LUT_DIR}/profiles.stamp)
endif()
add_custom_target(fe_luts ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${PROFILES_DIR}" "${BUNDLE_DIR}/Contents/Resources/profiles"
  DEPENDS ${PROFILES_STAMP}
  VERBATIM)

add_library(film_emulation_ofx SHARED plugin/ofx_plugin.cpp)
target_include_directories(film_emulation_ofx PRIVATE external/openfx/include)
target_link_libraries(film_emulation_ofx PRIVATE fe_engine)
set_target_properties(film_emulation_ofx PROPERTIES
  OUTPUT_NAME Fixer SUFFIX .ofx
  RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_DIR}/Contents/Win64
  LIBRARY_OUTPUT_DIRECTORY ${BUNDLE_DIR}/Contents/Win64
)
add_dependencies(film_emulation_ofx fe_luts)
add_custom_command(TARGET film_emulation_ofx POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/plugin/resources/com.alpcavas.fixer.png
          ${BUNDLE_DIR}/Contents/Resources/com.alpcavas.fixer.png)

add_executable(ofx_test_host engine/tools/ofx_test_host.cpp)
target_include_directories(ofx_test_host PRIVATE external/openfx/include)
target_link_libraries(ofx_test_host PRIVATE CUDA::cudart_static)

add_executable(fe_cli engine/tools/fe_cli.cpp)
target_link_libraries(fe_cli PRIVATE fe_engine)
set_target_properties(fe_cli PROPERTIES CUDA_RUNTIME_LIBRARY Static)
