#pragma once

#include "nvshmem.h"
#include "nvshmemx.h"

#define NVSHMEM_CHECK(stmt)                               \
 do {                                                    \
     int result = (stmt);                                \
     if (cudaSuccess != result) {                      \
         fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n",\
          __FILE__, __LINE__, result);                   \
         exit(-1);                                       \
     }                                                   \
 } while (0)

#define nvshmem_error_check(stmt) NVSHMEM_CHECK(stmt)