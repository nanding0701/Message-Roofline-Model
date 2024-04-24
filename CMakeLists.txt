##########################################################################################
#       CMake settings
##########################################################################################

# cmake min required
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(ex-msg-roofline)

# in source build warning
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    set(MSG "")
    message(STATUS "Warning! Building from the source directory is not recommended")
    message(STATUS "If unintented, please remove 'CMakeCache.txt' and 'CMakeFiles'")
    message(STATUS "and build from a separate directory")
    message(WARNING "In-source build")
endif()

# set cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/external/")

# set these as the defaults
set(CMAKE_ENABLE_EXPORTS ON CACHE BOOL "Executable exports symbols for loadable modules")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON CACHE BOOL "Append directories in the linker search path")

# Set a default build type if none was specified
set(EMR_BUILD_TYPE "RelWithDebInfo")

# set the build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${EMR_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${EMR_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

##########################################################################################
#       GCC version check
##########################################################################################
set(GCC_EXPECTED_VERSION 11.2)

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS GCC_EXPECTED_VERSION)
    message(FATAL_ERROR "GCC: GCB requires GCC v${GCC_EXPECTED_VERSION} or higher to build but found v${CMAKE_CXX_COMPILER_VERSION}")
endif()

##########################################################################################
#       CXX standard
##########################################################################################
set(CXX_STANDARD_REQUIRED ON)

# required minimum CXX standard
set(CMAKE_CXX_STANDARD_REQUIRED 17)

if(NOT CMAKE_CXX_STANDARD OR (CMAKE_CXX_STANDARD LESS ${CMAKE_CXX_STANDARD_REQUIRED}))
    set(CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD_REQUIRED})
    message(STATUS "Setting CXX_STANDARD to ${CMAKE_CXX_STANDARD_REQUIRED}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++${CMAKE_CXX_STANDARD}")
endif()

##########################################################################################
#       Check and set CMAKE_CUDA_HOST_COMPILER
##########################################################################################
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    message(STATUS "CMAKE_CUDA_HOST_COMPILER not provided\n=> Setting CMAKE_CUDA_HOST_COMPILER=$(which CC)")

    # run which CC to find the CUDA_HOST_COMPILER
    execute_process(
        COMMAND bash -c "which CC"
        OUTPUT_VARIABLE CUDA_HOST_COMPILER)

    # strip the last \n from the string
    string(STRIP ${CUDA_HOST_COMPILER} CUDA_HOST_COMPILER)

    # check for empty string
    if (NOT CUDA_HOST_COMPILER STREQUAL "")
        # set the CMAKE_CUDA_HOST_COMPILER and cache it for future use
        set(CMAKE_CUDA_HOST_COMPILER ${CUDA_HOST_COMPILER} CACHE STRING "CUDA_HOST_COMPILER" FORCE)
        message(STATUS "Setting CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}")
    else()
        message(FATAL_ERROR "CMAKE_CUDA_HOST_COMPILER not found. Please set it manually to CC or provide via CMAKE_CUDA_HOST_COMPILER")
    endif()

endif()

##########################################################################################
#       OpenMP
##########################################################################################
find_package(OpenMP REQUIRED)

if(OpenMP_FOUND)
    list(APPEND _OMP OpenMP::OpenMP_CXX)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CUDA_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

endif()

##########################################################################################
#       NVSHMEM settings
##########################################################################################
option(USE_NVSHMEM "Enable NVSHMEM.." ON)

if (DEFINED ENV{NVSHMEM_MPI_SUPPORT})
    set(NVSHMEM_MPI_SUPPORT_DEFAULT $ENV{NVSHMEM_MPI_SUPPORT})
else()
    set(NVSHMEM_MPI_SUPPORT_DEFAULT ON)
endif()

set(NVSHMEM_PREFIX $ENV{NVSHMEM_PREFIX} CACHE PATH "path to NVSHMEM install directory.")
set(CUDA_HOME $ENV{CUDA_HOME} CACHE PATH "path to CUDA installation")
set(MPI_HOME $ENV{MPI_HOME} CACHE PATH "path to MPI installation")
set(SHMEM_HOME ${MPI_HOME} CACHE PATH "path to SHMEM installation")
set(CUDA_ARCHITECTURES $ENV{CUDA_ARCHITECTURES} CACHE PATH "CUDA architectures to build for")

option(NVSHMEM_MPI_SUPPORT "Enable compilation of the MPI bootstrap and MPI-specific code" ${NVSHMEM_MPI_SUPPORT_DEFAULT})
option(NVSHMEM_SHMEM_SUPPORT "Enable Compilation of the SHMEM bootstrap and SHMEM specific code" $ENV{NVSHMEM_SHMEM_SUPPORT})
option(NVSHMEM_VERBOSE "Enable the ptxas verbose compilation option" OFF)

if (USE_NVSHMEM)
    find_package(NVSHMEM REQUIRED HINTS ${NVSHMEM_PREFIX}/lib/cmake/nvshmem)
endif()

##########################################################################################
#       Setup CUDA and MPI for aliases
##########################################################################################
if (USE_NVSHMEM)
    find_package(CUDAToolkit REQUIRED)
endif()
find_package(MPI REQUIRED)

if(NOT DEFINED CUDA_ARCHITECTURES)
    set(CUDA_ARCHITECTURES 80)
endif()

# set this variable
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

##########################################################################################
#       SHMEM support
##########################################################################################

if(NVSHMEM_SHMEM_SUPPORT)
    find_library(
      SHMEM_LIB
      NAMES oshmem
      HINTS ${SHMEM_HOME}
      PATH_SUFFIXES lib lib64)

    find_path(SHMEM_INCLUDE NAME shmem.h HINTS ${SHMEM_HOME}
              PATH_SUFFIXES include)

    add_library(shmem IMPORTED INTERFACE)
    target_link_libraries(shmem INTERFACE ${SHMEM_LIB})
    target_include_directories(shmem INTERFACE ${SHMEM_INCLUDE})

    if(NVSHMEM_MPI_SUPPORT)
          separate_arguments(SHMEM_C_LINK_FLAGS NATIVE_COMMAND "${MPI_C_LINK_FLAGS}")
          target_link_options(shmem INTERFACE ${SHMEM_C_LINK_FLAGS})
          target_compile_definitions(shmem INTERFACE ${MPI_C_COMPILE_DEFINITIONS})
          target_compile_options(shmem INTERFACE ${MPI_C_COMPILE_OPTIONS})
    endif()
endif()

##########################################################################################
#       Configure Header files
##########################################################################################
message(STATUS "Configuring...")

# configure the file: config.hpp
configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/config.hpp.in config.hpp @ONLY)

# install the configured file as well
install(FILES ${CMAKE_BINARY_DIR}/config.hpp DESTINATION include)

##########################################################################################
#       Add sub-directories
##########################################################################################

#----------------------------------------------------------------------------------------#
#   externals
#----------------------------------------------------------------------------------------#

#message(STATUS "Adding external...")
#add_subdirectory(external)

#----------------------------------------------------------------------------------------#
#   ex-msg-roofline source
#----------------------------------------------------------------------------------------#

message(STATUS "Adding ex-msg-roofline apps...")
add_subdirectory(apps)
