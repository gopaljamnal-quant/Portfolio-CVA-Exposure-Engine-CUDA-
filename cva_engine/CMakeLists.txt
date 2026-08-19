cmake_minimum_required(VERSION 3.18)
project(CVAExposureEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Target a broad but modern set of architectures. Override on the command
# line with -DCMAKE_CUDA_ARCHITECTURES=... for a specific GPU generation.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)
endif()

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Locate pybind11. Prefer an installed CMake package config; fall back to
# asking the active Python interpreter (works with a plain `pip install
# pybind11`, which is the common case in a fresh virtualenv/CI image).
# ---------------------------------------------------------------------------
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_CMAKEDIR_RESULT
    )
    if(NOT PYBIND11_CMAKEDIR_RESULT EQUAL 0)
        message(FATAL_ERROR "pybind11 not found. Install it with: pip install pybind11")
    endif()
    find_package(pybind11 CONFIG REQUIRED)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Core CUDA/C++ library: market model simulation, the host-side analytic
# Portfolio layer, and the GPU exposure/CVA engine.
# ---------------------------------------------------------------------------
add_library(cva_core STATIC
    src/MarketModels.cu
    src/Portfolio.cpp
    src/CVAEngine.cu
)

target_compile_features(cva_core PUBLIC cxx_std_17)
target_compile_options(cva_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
)
set_target_properties(cva_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(cva_core PUBLIC CUDA::cudart CUDA::curand)
target_include_directories(cva_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Python extension module
# ---------------------------------------------------------------------------
pybind11_add_module(cva_engine wrapper/pybind_wrapper.cpp)
target_link_libraries(cva_engine PRIVATE cva_core CUDA::cudart CUDA::curand)
set_target_properties(cva_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_VISIBILITY_PRESET hidden
)

install(TARGETS cva_engine LIBRARY DESTINATION .)

message(STATUS "CVA Exposure Engine configured. CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
