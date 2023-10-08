#include <torch/extension.h>
#include <mma.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_ARG(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_COMPATIBILITY(x, y, d1, d2) TORCH_CHECK_LINALG(\
    x.size(d1) == y.size(d2), #x " must be the same size by dim(" #d1 ") as " #y " by dim(" #d2 ")")

#define CHECK_DIM_FOR_WMMA(x, wmma_size, dim) TORCH_CHECK_VALUE( \
    x.size(dim) % wmma_size == 0, \
    #x " size by dim(" #dim ") must be multiples of " #wmma_size " for WMMA")


using namespace nvcuda;

using uint = __uint32_t;

template <typename scalar_t>
using accessor_1d = torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>;

template <typename scalar_t>
using accessor_2d = torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>;

template <uint wmma_m, uint wmma_n, uint wmma_k, typename layout_t>
using wmma_fragment_a = wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, half, layout_t>;

template <uint wmma_m, uint wmma_n, uint wmma_k, typename layout_t>
using wmma_fragment_b = wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, half, layout_t>;

template <uint wmma_m, uint wmma_n, uint wmma_k>
using wmma_fragment_c = wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, float>;


template <bool a_transposed = false, bool b_transposed = false>
__device__ __forceinline__ bool product_matrix_guard(
    const accessor_2d<c10::Half> matrix_a,
    const accessor_2d<c10::Half> matrix_b,
    const uint a_row_or_col,
    const uint b_col_or_row) {

    if constexpr (not a_transposed) {
        if (a_row_or_col >= matrix_a.size(0)) {
            return false;
        }
    } else {
        if (a_row_or_col >= matrix_a.size(1)) {
            return false;
        }
    }

    if constexpr (not b_transposed) {
        if (b_col_or_row >= matrix_b.size(1)) {
            return false;
        }
    } else {
        if (b_col_or_row >= matrix_b.size(0)) {
            return false;
        }
    }

    return true;
}


template <typename dst_scalar_t, bool transposed = false, typename src_scalar_t>
__device__ dst_scalar_t *get_fragment_ptr(
    const accessor_2d<src_scalar_t> matrix, uint row, uint col, uint ld) {

    if constexpr (transposed) {
        return reinterpret_cast<dst_scalar_t*>(matrix.data()) + ld * col + row;
    }

    return reinterpret_cast<dst_scalar_t*>(matrix.data()) + ld * row + col;
}


template <uint wmma_m, uint wmma_n, uint wmma_k, bool transpose_a, bool transpose_b>
__device__ __forceinline__ void matmul_wmma_helper(
    const accessor_2d<c10::Half> matrix_a,
    const accessor_2d<c10::Half> matrix_b,
    accessor_2d<float> matrix_c,
    const uint a_row_or_col,
    const uint b_col_or_row) {

    using layout_a = typename std::conditional<
        transpose_a, wmma::col_major, wmma::row_major>::type;
    
    using layout_b = typename std::conditional<
        transpose_b, wmma::col_major, wmma::row_major>::type;

    wmma_fragment_a<wmma_m, wmma_n, wmma_k, layout_a> a_frag;
    wmma_fragment_b<wmma_m, wmma_n, wmma_k, layout_b> b_frag;
    wmma_fragment_c<wmma_m, wmma_n, wmma_k> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    uint ld_a = matrix_a.size(1);
    uint ld_b = matrix_b.size(1);
    uint ld_c = matrix_c.size(1);

    uint sd_a = transpose_a ? matrix_a.size(0) : matrix_a.size(1);

    for (uint k = 0; k < sd_a; k += wmma_k) {
        half *ptr_a = get_fragment_ptr<half, transpose_a>(matrix_a, a_row_or_col, k, ld_a);
        half *ptr_b = get_fragment_ptr<half, transpose_b>(matrix_b, k, b_col_or_row, ld_b);

        wmma::load_matrix_sync(a_frag, ptr_a, ld_a);
        wmma::load_matrix_sync(b_frag, ptr_b, ld_b);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* ptr_c = get_fragment_ptr<float>(matrix_c, a_row_or_col, b_col_or_row, ld_c);
    wmma::store_matrix_sync(ptr_c, c_frag, ld_c, wmma::mem_row_major);
}


template <uint wmma_m, uint wmma_n, uint wmma_k>
__global__ void linear_fwd_kern_wmma(
    const accessor_2d<c10::Half> input,
    const accessor_2d<c10::Half> weight,
    const accessor_1d<c10::Half> bias,
    accessor_2d<float> output) {

    uint thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint warp_x = thread_x / warpSize;
    uint warp_y = blockIdx.y * blockDim.y + threadIdx.y;

    uint input_row = warp_y * wmma_m;
    uint weight_row = warp_x * wmma_n;

    if (not product_matrix_guard<false, true>(
        input, weight, input_row, weight_row)) {
        return;
    }

    matmul_wmma_helper<wmma_m, wmma_n, wmma_k, false, true>(
        input, weight, output, input_row, weight_row);

    uint warp_offset_x = warp_x * warpSize;
    uint thread_x_tile = thread_x - warp_offset_x;

    if (thread_x_tile < wmma_n) {
        #pragma unroll
        for (uint i = 0; i < wmma_m; i++) {
            output[input_row + i][weight_row + thread_x_tile] \
                += static_cast<float>(bias[weight_row + thread_x_tile]);
        }
    }
}


template <uint wmma_m, uint wmma_n, uint wmma_k>
__global__ void linear_bwd_kern_wmma(
    const accessor_2d<c10::Half> input,
    const accessor_2d<c10::Half> weight,
    const accessor_2d<c10::Half> d_output,
    accessor_2d<float> d_input,
    accessor_2d<float> d_weight,
    accessor_1d<float> d_bias) {

    uint thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint thread_y = blockIdx.y * blockDim.y + threadIdx.y;

    uint warp_x = thread_x / warpSize;
    uint warp_y = thread_y;

    uint a_row_or_col = warp_y * wmma_m;
    uint b_col_or_row = warp_x * wmma_n;

    /* dX = dY @ W */
    uint d_output_row = a_row_or_col;
    uint weight_col = b_col_or_row;

    if (product_matrix_guard<false, false>(
        d_output, weight, d_output_row, weight_col)) {

        matmul_wmma_helper<wmma_m, wmma_n, wmma_k, false, false>(
        d_output, weight, d_input, d_output_row, weight_col);
    }

    /* dW = dY^T @ X */
    uint d_output_col = a_row_or_col;
    uint input_col = b_col_or_row;

    if (product_matrix_guard<true, false>(
        d_output, input, d_output_col, input_col)) {

        matmul_wmma_helper<wmma_m, wmma_n, wmma_k, true, false>(
        d_output, input, d_weight, d_output_col, input_col);
    }

    /* db = SUM(dY, m) */
    if (thread_x < d_output.size(1) and a_row_or_col < d_output.size(0)) {
        float part = 0.0f;

        #pragma unroll
        for (uint i = 0; i < wmma_m; i++) {
            part += static_cast<float>(d_output[a_row_or_col + i][thread_x]);
        }

        atomicAdd(&d_bias[thread_x], part);
    }
}


__forceinline__ unsigned int div_and_ceil(float x, float y) {
    return ceil(x / y);
}


torch::Tensor linear_forward_fp16(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    CHECK_ARG(input);
    CHECK_ARG(weight);
    CHECK_ARG(bias);

    CHECK_COMPATIBILITY(input, weight, 1, 1);
    CHECK_COMPATIBILITY(bias, weight, 0, 0);

    constexpr dim3 wmma_dim = {
        16, // M
        16, // N
        16  // K
    };

    CHECK_DIM_FOR_WMMA(input, wmma_dim.x, 0);
    CHECK_DIM_FOR_WMMA(input, wmma_dim.z, 1);
    CHECK_DIM_FOR_WMMA(weight, wmma_dim.y, 0);
    CHECK_DIM_FOR_WMMA(weight, wmma_dim.z, 1);

    auto input_rows = input.size(0);
    auto weight_rows = weight.size(0);

    auto output = torch::zeros(
        {input_rows, weight_rows},
        input.options().dtype(torch::kFloat32)
    );

    const int warp_size = 32;
    const dim3 block_dim = {128, 4};

    auto m = input_rows;
    auto n = weight_rows;

    const dim3 grid_dim = {
        div_and_ceil(n * warp_size, wmma_dim.y * block_dim.x),
        div_and_ceil(m, wmma_dim.x * block_dim.y)
    };

    linear_fwd_kern_wmma<wmma_dim.x, wmma_dim.y, wmma_dim.z><<<grid_dim, block_dim>>>(
        input.packed_accessor32<c10::Half, 2, torch::RestrictPtrTraits>(),
        weight.packed_accessor32<c10::Half, 2, torch::RestrictPtrTraits>(),
        bias.packed_accessor32<c10::Half, 1, torch::RestrictPtrTraits>(),
        output.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    return output.toType(torch::kFloat16);
}


std::vector<torch::Tensor> linear_backward_fp16(
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

    constexpr dim3 wmma_dim = {
        16, // M
        16, // N
        16  // K
    };

    CHECK_DIM_FOR_WMMA(d_output, wmma_dim.x, 0);
    CHECK_DIM_FOR_WMMA(d_output, wmma_dim.z, 1);

    CHECK_DIM_FOR_WMMA(weight, wmma_dim.y, 1);
    CHECK_DIM_FOR_WMMA(weight, wmma_dim.z, 0);

    CHECK_DIM_FOR_WMMA(input, wmma_dim.y, 1);
    CHECK_DIM_FOR_WMMA(input, wmma_dim.z, 0);

    auto input_rows = input.size(0);
    auto input_cols = input.size(1);

    auto d_input = torch::zeros_like(
        input, input.options().dtype(torch::kFloat32));

    auto d_weight = torch::zeros_like(
        weight, weight.options().dtype(torch::kFloat32));

    auto d_bias = torch::zeros_like(
        bias, bias.options().dtype(torch::kFloat32));

    const int warp_size = 32;
    const dim3 block_dim = {128, 4};

    auto n = input_cols;
    auto m = std::max({input_cols, input_rows});

    const dim3 grid_dim = {
        div_and_ceil(n * warp_size, wmma_dim.y * block_dim.x),
        div_and_ceil(m, wmma_dim.x * block_dim.y)
    };

    linear_bwd_kern_wmma<wmma_dim.x, wmma_dim.y, wmma_dim.z><<<grid_dim, block_dim>>>(
        input.packed_accessor32<c10::Half, 2, torch::RestrictPtrTraits>(),
        weight.packed_accessor32<c10::Half, 2, torch::RestrictPtrTraits>(),
        d_output.packed_accessor32<c10::Half, 2, torch::RestrictPtrTraits>(),
        d_input.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        d_weight.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        d_bias.packed_accessor32<float, 1, torch::RestrictPtrTraits>()
    );

    return {
        d_input.toType(torch::kFloat16),
        d_weight.toType(torch::kFloat16), 
        d_bias.toType(torch::kFloat16)
    };
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward_fp16, "Custom linear layer (forward)");
    m.def("linear_backward", &linear_backward_fp16, "Custom linear layer (backward)");
}