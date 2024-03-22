#pragma once

#include <mpi.h>

#define MPI_CHECK(stmt)                                                                         \
    do {                                                                                        \
        int result = (stmt);                                                                    \
        if (MPI_SUCCESS != result) {                                                            \
            fprintf(stderr, "[%s:%d] MPI failed with error %d \n", __FILE__, __LINE__, result); \
            exit(-1);                                                                           \
        }                                                                                       \
    } while (0)

#define mpi_error_check(stmt) MPI_CHECK(stmt)