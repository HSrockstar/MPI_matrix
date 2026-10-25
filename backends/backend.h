#ifndef MATRIX_BACKEND_V1_H
#define MATRIX_BACKEND_V1_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Row-major A/B int32 in [-99,99], int64 result, no C epilogue.
 * add=0 overwrites all logical D elements, add=1 adds one K contribution.
 * m/k/n may be zero. No global result cache or application threads allowed.
 * Return nonzero with a NUL-terminated diagnostic; never throw across this ABI.
 */
typedef int (*matrix_backend_fn)(const int32_t *, const int32_t *, int64_t *, int, int,
                                 int, int, int, int, int, char *, size_t);
int matrix_backend_v1(const int32_t *, const int32_t *, int64_t *, int, int, int, int,
                      int, int, int, char *, size_t);
#ifdef __cplusplus
}
#endif
#endif
