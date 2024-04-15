project(mpi-apps LANGUAGES CXX)


file(GLOB CPP_SOURCES "*.cpp")

# C++ files
foreach(source_file ${CPP_SOURCES})

  # get the file name without an extension
  get_filename_component(exec_name ${source_file} NAME_WE)

  # add an executable with the same name as the source file
  add_executable(${exec_name} ${_EXCLUDE} ${source_file})

  # add dependency on argparse
  # add_dependencies(${exec_name} argparse)

  set_source_files_properties(${source_file} PROPERTIES LANGUAGE CXX
                                                        LINKER_LANGUAGE CXX)
  target_include_directories(
    ${exec_name}
    PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
            ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

  target_link_libraries(${exec_name} PUBLIC ${MPI_LIBS})

  set_target_properties(
    ${exec_name}
    PROPERTIES CXX_STANDARD ${CMAKE_CXX_STANDARD}
               CXX_EXTENSIONS NO
               INSTALL_RPATH_USE_LINK_PATH ON)

  # installation
  install(TARGETS ${exec_name} DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()

# C files

file(GLOB C_SOURCE "*.c")

foreach(source_file ${C_SOURCES})

  # get the file name without an extension
  get_filename_component(exec_name ${source_file} NAME_WE)

  # add an executable with the same name as the source file
  add_executable(${exec_name} ${_EXCLUDE} ${source_file})

  # add dependency on argparse
  # add_dependencies(${exec_name} argparse)

  set_source_files_properties(${source_file} PROPERTIES LANGUAGE C
                                                        LINKER_LANGUAGE C)
  target_include_directories(
    ${exec_name}
    PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
            ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

  target_link_libraries(${exec_name} PUBLIC ${MPI_LIBS})

  set_target_properties(
    ${exec_name}
    PROPERTIES CXX_STANDARD ${CMAKE_CXX_STANDARD}
               CXX_EXTENSIONS NO
               INSTALL_RPATH_USE_LINK_PATH ON)

  # installation
  install(TARGETS ${exec_name} DESTINATION ${CMAKE_INSTALL_BINDIR})
endforeach()

# CUDA files
if (USE_NVSHMEM)

  file(GLOB CUDA_SOURCES "*.cu")

  # CUDA files
  foreach(source_file ${CUDA_SOURCES})

    # doesn't matter if we do this for each file.
    # but don't do it outside if not needed.
    enable_language(CUDA)

    # get the file name without an extension
    get_filename_component(exec_name ${source_file} NAME_WE)

    # add an executable with the same name as the source file
    add_executable(${exec_name} ${_EXCLUDE} ${source_file})

    # add dependency on argparse
    # add_dependencies(${exec_name} argparse)

    set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA
                                                          LINKER_LANGUAGE CUDA)
    target_include_directories(
      ${exec_name}
      PRIVATE ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}/../../include
              ${ARGPARSE_INCLUDE_DIR} ${MDSPAN_INCLUDE_DIR})

    target_link_libraries(${exec_name} PUBLIC ${MPI_LIBS})

    set_target_properties(
      ${exec_name}
      PROPERTIES
      CXX_STANDARD ${CMAKE_CXX_STANDARD}
      CXX_EXTENSIONS ON
      CUDA_ARCHITECTURES 80
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO
      INSTALL_RPATH_USE_LINK_PATH ON)

    # installation
    install(TARGETS ${exec_name} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endforeach()

endif()
