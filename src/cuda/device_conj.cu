#include "blas/device.hh"
#include "thrust/complex.h"

#if defined(BLAS_HAVE_CUBLAS)

namespace blas {

template <typename scalar_t>
__global__ void conj_kernel(
    int64_t n,
    scalar_t* src, int64_t inc_src,
    scalar_t* dst, int64_t inc_dst)
{
    using thrust::conj;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[ i*inc_dst ] = conj( src[ i*inc_src ] );
}

//------------------------------------------------------------------------------
/// Conjugates each element of the vector src and stores in dst.
///
/// @param[in] n
///     Number of elements in the vector. n >= 0.
///
/// @param[in] src
///     Pointer to the input vector of length n.
///
/// @param[in] inc_src
///     Stride between elements of src. inc_src >= 1.
///
/// @param[out] dst
///     Pointer to output vector
///     On exit, each element dst[i] is updated as dst[i] = conj( src[i] ).
//      dst may be the same as src.
///
/// @param[in] inc_dst
///     Stride between elements of dst. inc_dst >= 1.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void conj(
    int64_t n,
    scalar_t* src, int64_t inc_src,
    scalar_t* dst, int64_t inc_dst,
    blas::Queue& queue )
{
    if (n <= 0) {
        return;
    }

    const int BlockSize = 128;

    int64_t n_threads = std::min( int64_t( BlockSize ), n );
    int64_t n_blocks = 1 + ((n - 1) / n_threads);

    blas_dev_call(
        cudaSetDevice( queue.device() ) );

    conj_kernel<<<n_blocks, n_threads, 0, queue.stream()>>>(
        n, src, inc_src, dst, inc_dst );

    blas_dev_call(
        cudaGetLastError() );
}

} // namespace blas

#endif // BLAS_HAVE_CUBLAS
