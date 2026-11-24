cmake_minimum_required(VERSION 3.18)
project(openfocus_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
  RESULT_VARIABLE _nanobind_cmake_res
)
if(NOT _nanobind_cmake_res EQUAL 0)
  message(FATAL_ERROR "Failed to locate nanobind CMake directory via ${Python_EXECUTABLE}")
endif()

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  _openfocus_native
  NB_STATIC
  cpp/module.cpp
  cpp/ecc_solver.cpp
  cpp/openfocus_ab.cpp
  cpp/openfocus_c.cpp
)

target_include_directories(_openfocus_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cpp")

if(MSVC)
  target_compile_options(_openfocus_native PRIVATE /O2 /fp:precise /EHsc)
else()
  target_compile_options(_openfocus_native PRIVATE -O3 -fno-fast-math)
  # The pyramid kernels round each multiplication and addition separately.
  set_property(SOURCE cpp/openfocus_c.cpp APPEND PROPERTY COMPILE_OPTIONS -ffp-contract=off)
endif()

if(MINGW)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # Older CPython headers select the 32-bit object layout under MinGW.
    target_compile_definitions(nanobind-static PUBLIC "MS_WIN64=")
  endif()
  target_link_options(
    _openfocus_native
    PRIVATE
    -static
    -static-libgcc
    -static-libstdc++
  )
endif()

install(TARGETS _openfocus_native LIBRARY DESTINATION core)

if(SKBUILD)
  install(FILES constants.py build_identity.py DESTINATION .)
endif()
