cmake_minimum_required(VERSION 3.24)

# Set CUDA architectures BEFORE project() so they take effect
set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures (RTX 4090)")

project(fish-audio-cpp VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit 12.4 REQUIRED)

# TODO: Uncomment when dev packages are installed
# find_package(sentencepiece REQUIRED)
# find_package(CURL REQUIRED)
# find_package(SampleRate REQUIRED)

# FlashAttention
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
# TODO: Uncomment when FlashAttention headers/libs are available
# find_package(FlashAttention REQUIRED)

find_library(CUDNN_LIBRARY cudnn PATHS ${CUDAToolkit_LIBRARY_DIR} /usr/lib/x86_64-linux-gnu REQUIRED)
find_path(CUDNN_INCLUDE_DIR cudnn.h
  PATHS ${CUDAToolkit_INCLUDE_DIRS} /usr/include /usr/include/x86_64-linux-gnu REQUIRED)

# nlohmann/json, spdlog, cpp-httplib, cxxopts, gtest via FetchContent
include(FetchContent)
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3)
FetchContent_MakeAvailable(json)

FetchContent_Declare(spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.14.1)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(cpp-httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.18.1)
FetchContent_MakeAvailable(cpp-httplib)

FetchContent_Declare(cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG v3.2.1)
FetchContent_MakeAvailable(cxxopts)

FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(sentencepiece
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG v0.2.0)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sentencepiece)

# Core library (static lib for engine + model + server)
add_library(fish_core STATIC
  src/model/loader.cc
  src/model/dual_ar_config.cc
  src/model/dac_config.cc
  src/tokenizer/tokenizer.cc
  src/engine/block_manager.cc
  src/engine/dual_ar_engine.cc
  src/engine/dac_engine.cc
  src/engine/scheduler.cc
  src/engine/inference_pipeline.cc
  src/server/http_server.cc
  src/server/routes.cc
)

# CUDA kernel library
add_library(fish_kernels STATIC
  src/kernels/paged_attention.cu
  src/kernels/rms_norm.cu
  src/kernels/rope.cu
  src/kernels/ras_sampling.cu
  src/kernels/sampling.cu
  src/kernels/snake.cu
  src/kernels/rvq_lookup.cu
  src/kernels/ffn_activations.cu
  src/kernels/embedding.cu
  src/kernels/prefill_attention.cu
)

target_include_directories(fish_kernels PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(fish_kernels PUBLIC
  spdlog::spdlog
)

target_include_directories(fish_core PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${sentencepiece_SOURCE_DIR}/src
  ${CUDNN_INCLUDE_DIR}
  # ${FlashAttention_INCLUDE_DIR}
)

target_link_libraries(fish_core PUBLIC
  CUDA::cublas
  CUDA::cuda_driver
  CUDA::cudart_static
  ${CUDNN_LIBRARY}
  # ${FlashAttention_LIBRARY}
  sentencepiece
  # SampleRate::samplerate
  nlohmann_json::nlohmann_json
  spdlog::spdlog
  httplib::httplib
  fish_kernels
)

# Main executable
add_executable(fish-server src/main.cc)
target_link_libraries(fish-server PRIVATE fish_core httplib::httplib cxxopts::cxxopts)

# Tests
enable_testing()
add_executable(test_fish
  tests/test_kernels.cu
  tests/test_block_manager.cc
  tests/test_dual_ar_engine.cc
  tests/test_dac_engine.cc
  tests/test_scheduler.cc
  tests/test_integration.cc
  tests/test_model_loader.cc
)
target_link_libraries(test_fish PRIVATE fish_core GTest::gtest GTest::gtest_main)
add_test(NAME fish_tests COMMAND test_fish)
