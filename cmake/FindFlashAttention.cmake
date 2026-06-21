# cmake/FindFlashAttention.cmake
find_path(FlashAttention_INCLUDE_DIR
  NAMES flash_attn.h
  PATHS
    /usr/local/flash_attn/include
    ${CMAKE_SOURCE_DIR}/third_party/flash-attention/csrc
    ${CMAKE_SOURCE_DIR}/third_party/flash-attention/hopper
)

find_library(FlashAttention_LIBRARY
  NAMES flash_attn
  PATHS
    /usr/local/flash_attn/lib
    ${CMAKE_SOURCE_DIR}/third_party/flash-attention/build
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FlashAttention
  REQUIRED_VARS FlashAttention_LIBRARY FlashAttention_INCLUDE_DIR)

if(FlashAttention_FOUND)
  add_library(FlashAttention::flash_attn UNKNOWN IMPORTED)
  set_target_properties(FlashAttention::flash_attn PROPERTIES
    IMPORTED_LOCATION "${FlashAttention_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FlashAttention_INCLUDE_DIR}")
endif()
