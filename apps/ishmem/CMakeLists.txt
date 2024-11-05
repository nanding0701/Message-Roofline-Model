# each .cpp file is a new target
file(GLOB ISHMEM_APP_SOURCES "*.cpp")

# loop through all source files
foreach(source_file ${ISHMEM_APP_SOURCES})

    # get file name
    get_filename_component(NAME_ ${source_file} NAME_WE)

    # add a new target
    add_executable(${NAME_} ${source_file})

    # link and include dependency libraries 
    target_link_libraries(${NAME_} PRIVATE SHMEM)
    target_link_libraries(${NAME_} PRIVATE ISHMEM)
    target_include_directories(${NAME_} PRIVATE SHMEM)
    target_include_directories(${NAME_} PRIVATE ISHMEM)
    target_link_options(${NAME_} PRIVATE -lze_loader)

    # install
    install(TARGETS ${NAME_} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

endforeach()
