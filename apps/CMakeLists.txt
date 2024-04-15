# ----------------------------------------------------------------------------------------#
# Add nvshmem apps
# ----------------------------------------------------------------------------------------#

if (USE_NVSHMEM)
    message(STATUS "Adding nvshmem apps...")
    add_subdirectory(nvshmem)
endif()

# ----------------------------------------------------------------------------------------#
# Add mpi apps
# ----------------------------------------------------------------------------------------#

message(STATUS "Adding mpi apps...")
add_subdirectory(mpi)

message(STATUS "Adding rocshmem apps...")
add_subdirectory(rocshmem)
