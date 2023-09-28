#include <torch/extension.h>
#include <mma.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_ARG(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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


template <bool a_transposed, bool b_transposed>
__device__ bool matrix_guard(
    const accessor_2d<c10::Half> matrix_a,
    const accessor_2d<c10::Half> matrix_b,
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


template <typename dst_scalar_t, bool transposed = false, typename src_scalar_t>
__device__ dst_scalar_t *get_fragment_ptr(
    const accessor_2d<src_scalar_t> matrix, uint row, uint col, uint ld) {

    if (transposed) {
        return reinterpret_cast<dst_scalar_t*>(matrix.data()) + ld * col + row;
    }

    return reinterpret_cast<dst_scalar_t*>(matrix.data()) + ld * row + col;
}


template <uint wmma_m, uint wmma_n, uint wmma_k, bool transpose_a, bool transpose_b>
__device__ void matmul_wmma(
    const accessor_2d<c10::Half> matrix_a,
    const accessor_2d<c10::Half> matrix_b,
    accessor_2d<float> matrix_c) {

    using layout_a = typename std::conditional<transpose_a, wmma::col_major, wmma::row_major>::type;
    using layout_b = typename std::conditional<transpose_b, wmma::col_major, wmma::row_major>::type;

    wmma_fragment_a<wmma_m, wmma_n, wmma_k, layout_a> a_frag;
    wmma_fragment_b<wmma_m, wmma_n, wmma_k, layout_b> b_frag;
    wmma_fragment_c<wmma_m, wmma_n, wmma_k> c_frag;

    uint warp_x = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    uint warp_y = (blockIdx.y * blockDim.y + threadIdx.y);

    uint a_row_or_col = warp_y * wmma_m;
    uint b_col_or_row = warp_x * wmma_n;

    if (not matrix_guard<transpose_a, transpose_b>(
        matrix_a, matrix_b, a_row_or_col, b_col_or_row)) {
        return;
    }

    wmma::fill_fragment(c_frag, 0.0f);

    uint ld_a = matrix_a.size(1);
    uint ld_b = matrix_b.size(1);
    uint ld_c = matrix_c.size(1);

    uint sd_a = transpose_a ? matrix_a.size(0) : matrix_a.size(1);

    for (int k = 0; k < sd_a; k += wmma_k) {
        half *ptr_a = get_fragment_ptr<half, transpose_a>(matrix_a, a_row_or_col, k, ld_a);
        half *ptr_b = get_fragment_ptr<half, transpose_b>(matrix_b, k, b_col_or_row, ld_b);

        wmma::load_matrix_sync(a_frag, ptr_a, ld_a);
        wmma::load_matrix_sync(b_frag, ptr_b, ld_b);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* ptr_c = get_fragment_ptr<float>(matrix_c, a_row_or_col, b_col_or_row, ld_c);
    wmma::store_matrix_sync(ptr_c, c_frag, ld_c, wmma::mem_row_major);
}


template <uint wmma_m, uint wmma_n, uint wmma_k, bool transpose_a, bool transpose_b>
__global__ void matmul_wmma_kernel(
    const accessor_2d<c10::Half> matrix_a,
    const accessor_2d<c10::Half> matrix_b,
    accessor_2d<float> matrix_c) {

    matmul_wmma<wmma_m, wmma_n, wmma_k, transpose_a, transpose_b>(matrix_a, matrix_b, matrix_c);
}


__forceinline__ uint div_and_ceil(float x, float y) {
    return ceil(x / y);
}


template <bool transpose_a, bool transpose_b>
torch::Tensor matmul(
    torch::Tensor matrix_a,
    torch::Tensor matrix_b) {

    CHECK_ARG(matrix_a);
    CHECK_ARG(matrix_b);

    constexpr dim3 wmma_dim = {
        16, // M
        16, // N
        16  // K
    };

    auto m = transpose_a ? matrix_a.size(1) : matrix_a.size(0);
    auto n = transpose_b ? matrix_b.size(0) : matrix_b.size(1);

    auto matrix_c = torch::zeros(
        {m, n},
        matrix_a.options().dtype(torch::kFloat32)
    );

    const int warp_size = 32;
    const dim3 block_dim = {128, 4};

    dim3 grid_dim = {
        div_and_ceil(n * warp_size, wmma_dim.y * block_dim.x),
        div_and_ceil(m, wmma_dim.x * block_dim.y)
    };

    const auto accessor_matrix_a = matrix_a.packed_accessor32<c10::Half, 2, torch::RestrictPtrTraits>();
    const auto accessor_matrix_b = matrix_b.packed_accessor32<c10::Half, 2, torch::RestrictPtrTraits>();
    auto accessor_matrix_c = matrix_c.packed_accessor32<float, 2, torch::RestrictPtrTraits>();

    matmul_wmma_kernel<wmma_dim.x, wmma_dim.y, wmma_dim.z, transpose_a, transpose_b>
        <<<grid_dim, block_dim>>>(accessor_matrix_a, accessor_matrix_b, accessor_matrix_c);

    return matrix_c.toType(torch::kFloat16);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul<false, false>, "Performs C = AB with CUDA Tensor Cores");
    m.def("matmul_ta", &matmul<true, false>, "Performs C = A^TB with CUDA Tensor Cores");
    m.def("matmul_tb", &matmul<false, true>, "Performs C = AB^T with CUDA Tensor Cores");
    m.def("matmul_tatb", &matmul<true, true>, "Performs C = A^TB^T with CUDA Tensor Cores");
}