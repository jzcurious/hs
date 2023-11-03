#include <torch/extension.h>


using uint = __uint32_t;


template<typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t value) {
    for (uint offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(warpSize - 1, value, offset);
    }
    return value;
}


template<uint block_size, typename scalar_t>
__device__ __forceinline__ scalar_t block_reduce_sum(scalar_t value) {
    constexpr uint warp_num = block_size / 32;
    __shared__ scalar_t partial_sums[warp_num];

    uint wid = threadIdx.x / warpSize;
    uint lane = threadIdx.x % warpSize;

    value = warp_reduce_sum(value);

    if (lane == 0) {
        partial_sums[wid] = value;
    }

    __syncthreads();

    if (wid == 0) {
        value = (lane < warp_num) ? partial_sums[lane] : 0;
        value = warp_reduce_sum(value);
    }

    return value;
}


template<uint block_size, typename scalar_t>
__global__ void reduce_kernel(scalar_t* data, scalar_t* total_sum, uint numel) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    auto partial_sum = block_reduce_sum<block_size>(
        (tid < numel) ? data[tid] : 0
    );

    __syncthreads();

    if (tid < numel and threadIdx.x == 0) {
        atomicAdd(total_sum, partial_sum);
    }
}


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_ARG(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


__forceinline__ unsigned int div_and_ceil(float x, float y) {
    return ceil(x / y);
}


torch::Tensor reduce_sum(torch::Tensor input) {
    CHECK_ARG(input);

    auto output = torch::zeros(1, input.options());

    constexpr dim3 block_dim = {32 * 4, 1, 1};

    const dim3 grid_dim = {
        div_and_ceil(input.numel(), block_dim.x), 1, 1
    };

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "reduce_sum",
        ([&] {
            reduce_kernel<block_dim.x><<<grid_dim, block_dim>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                input.numel()
            );
        })
    );

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_sum", &reduce_sum, "Custom implementation of reducing summation (CUDA enabled)");
}