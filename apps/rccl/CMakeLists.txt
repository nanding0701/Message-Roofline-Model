cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)

###############################################################################
# GLOBAL COMPILE FLAGS
###############################################################################
if (NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER hipcc)
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
project(rccl_apps VERSION 1.0.0 LANGUAGES CXX)

###############################################################################
# ROCSHMEM DEPENDENCY
###############################################################################
find_package(hip REQUIRED)
find_package(rccl CONFIG REQUIRED)
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

  target_include_directories(
    ${NAME_}
    PRIVATE
      ${CMAKE_CURRENT_LIST_DIR}/../../include
  )

  target_link_libraries(
    ${NAME_}
    PRIVATE
      rccl
      MPI::MPI_CXX
      -fgpu-rdc
  )
endforeach()
