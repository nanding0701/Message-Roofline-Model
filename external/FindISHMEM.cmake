find_library(ishmem_lib_found ishmem PATHS ${ISHMEM_ROOT} SUFFIXES lib lib64 NO_DEFAULT_PATHS)
find_path(ishmem_headers_found ishmem.h PATHS ${ISHMEM_ROOT}/include NO_DEFAULT_PATHS)

find_package_handle_standard_args(ISHMEM DEFAULT_MSG ishmem_lib_found ishmem_headers_found)

if (ishmem_lib_found AND ishmem_headers_found)
  add_library(ISHMEM INTERFACE)
  set_target_properties(ISHMEM PROPERTIES
    INTERFACE_LINK_LIBRARIES ${ishmem_lib_found}
    INTERFACE_INCLUDE_DIRECTORIES ${ishmem_headers_found}
  )
endif()
