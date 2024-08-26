#pragma once

#include <rccl/rccl.h>

#define NCCL_CALL(call)                                                                     \
    {                                                                                       \
        ncclResult_t  ncclStatus = call;                                                    \
        if (ncclSuccess != ncclStatus) {                                                    \
            fprintf(stderr,                                                                 \
                    "ERROR: NCCL call \"%s\" in line %d of file %s failed "                 \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, ncclGetErrorString(ncclStatus), ncclStatus); \
            exit( ncclStatus );                                                             \
        }                                                                                   \
    }