#include <torch/extension.h>


using uint = __uint32_t;

template <typename scalar_t>
using accessor_1d = torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>;

template <typename scalar_t>
using accessor_2d = torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>;


template <typename scalar_t>
__forceinline__ __device__
void gpuAtomicAdd(scalar_t *acc_ptr, scalar_t part_val) {
    #if __CUDA_ARCH__ >= 700
        if constexpr (std::is_same_v<scalar_t, c10::Half>) {
            atomicAdd(
                reinterpret_cast<half*>(acc_ptr), 
                static_cast<half>(part_val)
            );
        }
        else {
            atomicAdd(acc_ptr, part_val);
        }
    #else
        if constexpr (std::is_same_v<scalar_t, float>) {
            atomicAdd(acc_ptr, part_val);
        }
        else {
            assert(false && "Not supported CUDA device.");
        }
    #endif
}


template <bool a_transposed = false, bool b_transposed = false, typename scalar_t>
__device__ __forceinline__ bool matrix_guard(
    const accessor_2d<scalar_t> matrix_a,
    const accessor_2d<scalar_t> matrix_b,
    const uint a_row_or_col,
    const uint b_col_or_row) {

    if ((not a_transposed) and a_row_or_col >= matrix_a.size(0)) {
        return false;
    }

    if ((not b_transposed) and b_col_or_row >= matrix_b.size(1)) {
        return false;
    }

    if (a_transposed and a_row_or_col >= matrix_a.size(1)) {
        return false;
    }

    if (b_transposed and b_col_or_row >= matrix_b.size(0)) {
        return false;
    }

    return true;
}


template <bool transposed = false, typename scalar_t>
__device__ __forceinline__ scalar_t get_matrix_elem(
    const accessor_2d<scalar_t> matrix, uint row, uint col) {

    if constexpr (transposed) {
        return matrix[col][row];
    } else {
        return matrix[row][col];
    }
}


template <uint tile_size, bool transpose_a, bool transpose_b, typename scalar_t>
__device__ void matmul_smma_helper(
    const accessor_2d<scalar_t> matrix_a,
    const accessor_2d<scalar_t> matrix_b,
    accessor_2d<scalar_t> matrix_c,
    uint m, uint n) {

    __shared__ scalar_t matrix_a_frag[tile_size][tile_size];
    __shared__ scalar_t matrix_b_frag[tile_size][tile_size];

    scalar_t acc = 0;

    auto sd = transpose_a ? matrix_a.size(0) : matrix_a.size(1);

    for (uint t = 0; t < sd; t += tile_size) {
        uint j = threadIdx.x;
        uint i = threadIdx.y;

        matrix_a_frag[i][j] = get_matrix_elem<transpose_a>(
            matrix_a, tile_size * blockIdx.y + i, t + j);
        
        matrix_b_frag[i][j] = get_matrix_elem<transpose_b>(
            matrix_b, t + i, tile_size * blockIdx.x + j);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < tile_size; k++) {
            acc += matrix_a_frag[i][k] * matrix_b_frag[k][j];
        }

        __syncthreads();
    }

    matrix_c[m][n] = acc;
}


template <uint tile_size, typename scalar_t>
__global__ void linear_fwd_kern_smem_g2d(
    const accessor_2d<scalar_t>input,
    const accessor_2d<scalar_t>weight,
    const accessor_1d<scalar_t>bias,
    accessor_2d<scalar_t> output) {

    __shared__ scalar_t local_bias[tile_size];

    auto n = blockIdx.x * blockDim.x + threadIdx.x;
    auto m = blockIdx.y * blockDim.y + threadIdx.y;

    if (not matrix_guard<false, true>(input, weight, m, n)) {
        return;
    }

    matmul_smma_helper<tile_size, false, true>(input, weight, output, m, n);

    if (threadIdx.y == 0) {
        local_bias[threadIdx.x] = bias[n];
    }

    __syncthreads();

    output[m][n] += local_bias[threadIdx.x];
}


template <uint tile_size, typename scalar_t>
__global__ void linear_bwd_kern_smem_g2d(
    const accessor_2d<scalar_t> input,
    const accessor_2d<scalar_t> weight,
    const accessor_2d<scalar_t> d_output,
    accessor_2d<scalar_t> d_input,
    accessor_2d<scalar_t> d_weight,
    accessor_1d<scalar_t> d_bias) {

    auto n = blockIdx.x * blockDim.x + threadIdx.x;
    auto m = blockIdx.y * blockDim.y + threadIdx.y;

    /* dX = dY @ W */
    if (matrix_guard<false, false>(d_output, weight, m, n)) {
        matmul_smma_helper<tile_size, false, false>(d_output, weight, d_input, m, n);
    }

    /* dW = dY^T @ X */
    if (matrix_guard<true, false>(d_output, input, m, n)) {
        matmul_smma_helper<tile_size, true, false>(d_output, input, d_weight, m, n);
    }

    /* db = SUM(dY, m) */
    if (m < d_output.size(0) and n < d_output.size(1)) {
        gpuAtomicAdd(&d_bias[n], d_output[m][n]);
    }
}


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_ARG(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_COMPATIBILITY(x, y, d1, d2) \
    TORCH_CHECK(x.size(d1) == y.size(d2), \
    #x " must be the same size by dim(" #d1 ") as " #y " by dim(" #d2 ")")


__forceinline__ unsigned int div_and_ceil(float x, float y) {
    return ceil(x / y);
}


torch::Tensor linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    CHECK_ARG(input);
    CHECK_ARG(weight);
    CHECK_ARG(bias);

    CHECK_COMPATIBILITY(input, weight, 1, 1);
    CHECK_COMPATIBILITY(bias, weight, 0, 0);

    auto batch_size = input.size(0);
    auto weight_rows = weight.size(0);

    auto output = torch::zeros(
        {batch_size, weight_rows},
        input.options()
    );

    constexpr dim3 block_dim = {16, 16};

    const dim3 grid_dim = {
        div_and_ceil(weight_rows, block_dim.x),
        div_and_ceil(batch_size, block_dim.y)
    };

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "linear_forward",
        ([&] {
            linear_fwd_kern_smem_g2d<block_dim.x><<<grid_dim, block_dim>>>(
                input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                weight.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
            );
        })
    );

    return output;
}


std::vector<torch::Tensor> linear_backward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor d_output) {

    CHECK_ARG(input);
    CHECK_ARG(weight);
    CHECK_ARG(bias);
    CHECK_ARG(d_output);

    CHECK_COMPATIBILITY(input, weight, 1, 1);
    CHECK_COMPATIBILITY(bias, weight, 0, 0);
    CHECK_COMPATIBILITY(d_output, weight, 1, 0);
    CHECK_COMPATIBILITY(d_output, bias, 1, 0);

    auto batch_size = input.size(0);
    auto weight_rows = weight.size(0);
    auto weight_cols = weight.size(1);

    auto d_input = torch::zeros_like(input);
    auto d_weight = torch::zeros_like(weight);
    auto d_bias = torch::zeros_like(bias);

    constexpr dim3 block_dim = {16, 16};

    const dim3 grid_dim = {
        div_and_ceil(weight_cols, block_dim.x),
        div_and_ceil(std::max({weight_cols, batch_size}), block_dim.y)
    };

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "linear_backward",
        ([&] {
            linear_bwd_kern_smem_g2d<block_dim.x><<<grid_dim, block_dim>>>(
                input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                weight.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                d_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                d_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                d_weight.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                d_bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
            );
        })
    );

    return {d_input, d_weight, d_bias};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "Custom linear layer (forward)");
    m.def("linear_backward", &linear_backward, "Custom linear layer (backward)");
}