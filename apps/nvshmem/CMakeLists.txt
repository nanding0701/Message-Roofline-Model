project(nvshmem-apps LANGUAGES CUDA CXX)

# check appropriate cuda compiler flags before building nvshmem apps
include(CheckCompilerFlag)
check_compiler_flag(CUDA -t4 NVCC_THREADS)

add_library(nvshmem ALIAS nvshmem::nvshmem)
add_library(nvshmem_host ALIAS nvshmem::nvshmem_host)
add_library(nvshmem_device ALIAS nvshmem::nvshmem_device)

# add an interface of libraries to link with tagets
add_library(nvshmem_helper INTERFACE)
target_link_libraries(nvshmem_helper INTERFACE CUDA::cudart CUDA::cuda_driver nvidia-ml cuda nvshmem_host nvrtc)

# add -lnvshmem static or dynamic libraries based on build configuration
if(NVSHMEM_TEST_STATIC_LIB)
    target_link_libraries(nvshmem_helper INTERFACE nvshmem)
else()
    target_link_libraries(nvshmem_helper INTERFACE nvshmem_host nvshmem_device)
endif()

# set appropriate c++ and cuda standards
target_compile_features(nvshmem_helper INTERFACE cxx_std_17 cuda_std_11)

# add OpenSHMEM libraries if needed
if(NVSHMEM_SHMEM_SUPPORT)
    target_link_libraries(nvshmem_helper INTERFACE shmem)
    target_compile_definitions(nvshmem_helper INTERFACE NVSHMEMTEST_SHMEM_SUPPORT)
endif()

# each .cu file is a new target
file(GLOB CUDA_SOURCES "*.cu")

# looop through all source files
foreach(source_file ${CUDA_SOURCES})

    # get file name
    get_filename_component(NAME_ ${source_file} NAME_WE)

    # add a new target
    add_executable(${NAME_} ${source_file})

    # link libraries from the nvshmem_helper interface
    target_link_libraries(${NAME_} nvshmem_helper)

    # add custom include directories
    target_include_directories(${NAME_} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

    # important: always add ${nvshmem_prefix}/lib link directory to avoid link time errors
    # target_link_directories(${NAME_} BEFORE PRIVATE ${NVSHMEM_PREFIX}/lib)

    # set all required target properties
    set_target_properties(${NAME_} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CUDA_STANDARD 11
    CUDA_SEPARABLE_COMPILATION ON
    )

    # compile with options
    target_compile_options(${NAME_}
    PRIVATE $<$<CONFIG:Debug>:-O0;-g;>
    $<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-g;-G>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
    )

    # install
    install(TARGETS ${NAME_} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

endforeach()