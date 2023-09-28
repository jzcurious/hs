#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_ARG(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using namespace nvcuda;

using uint = __uint32_t;

template <typename scalar_t, uint ndim>
using accessor = torch::PackedTensorAccessor32<scalar_t, ndim, torch::RestrictPtrTraits>;

template <typename func_t, typename scalar_t>
__global__ void map_kernel(func_t func, accessor<scalar_t>) {
    // TODO
}