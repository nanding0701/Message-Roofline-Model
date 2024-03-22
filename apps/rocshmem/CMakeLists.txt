###############################################################################
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)

###############################################################################
# GLOBAL COMPILE FLAGS
###############################################################################
if (NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
endif()
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")

###############################################################################
# DEFAULT BUILD TYPE
###############################################################################
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "CMAKE_BUILD_TYPE unspecified: generating RelWithDebInfo build")

  set(
    CMAKE_BUILD_TYPE
    "RelWithDebInfo"
    CACHE
      STRING
        "build type: Release, Debug, RelWithDebInfo, MinSizeRel"
    FORCE
  )
endif()

###############################################################################
# PROJECT
###############################################################################
project(rocshmem_apps VERSION 1.0.0 LANGUAGES CXX)

###############################################################################
# ROCSHMEM DEPENDENCY
###############################################################################
find_package(hip REQUIRED)
find_package(rocshmem CONFIG REQUIRED)
find_package(MPI REQUIRED)

###############################################################################
# SOURCES
###############################################################################

# each .cpp file is a new target
file(GLOB CXX_SOURCES "*.cpp")

# loop through all source files
foreach(source_file ${CXX_SOURCES})

  # get file name
  get_filename_component(NAME_ ${source_file} NAME_WE)

  # add a new target
  add_executable(${NAME_} ${source_file})

  target_include_directories(
    ${NAME_}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/../../library/src
  )

  set_target_properties(${NAME_} PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD_REQUIRED ON
  HIP_STANDARD_REQUIRED ON
  CXX_STANDARD 17
  HIP_SEPARABLE_COMPILATION ON
  )
  #target_sources(
  #  ${NAME_}
  #  PRIVATE
  #    hashtable_rocshmem.cpp
  #)

  target_include_directories(
    ${NAME_}
    PRIVATE
      rocshmem::rocshmem
      ${CMAKE_CURRENT_LIST_DIR}/../../include
  )

  target_link_libraries(
    ${NAME_}
    PRIVATE
      rocshmem::rocshmem
      hip::host
      MPI::MPI_CXX
      -fgpu-rdc
  )
endforeach()
