#include <mma.h>
#include <torch/extension.h>

using namespace nvcuda;


template <int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void kernel_matmul_t_wmma(half *a, half *b, float *c, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_t_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    int warp_x = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warp_y = (blockIdx.y * blockDim.y + threadIdx.y);

    int a_row = warp_y * WMMA_M;
    int b_t_col = warp_x * WMMA_N;

    if (a_row >= M or b_t_col >= N) {
        return;
    }

    wmma::fill_fragment(c_frag, 0.0f);

    int lda = K;
    int ldb = K;
    int ldc = N;

    for (int k = 0; k < K; k += WMMA_K) {
        int a_col = k;
        int b_t_row = k;

        half* ptr_a = a + lda * a_row + a_col;
        half* ptr_b = b + ldb * b_t_col + b_t_row;

        wmma::load_matrix_sync(a_frag, ptr_a, lda);
        wmma::load_matrix_sync(b_t_frag, ptr_b, ldb);

        wmma::mma_sync(c_frag, a_frag, b_t_frag, c_frag);
    }

    float* ptr_c = c + ldc * a_row + b_t_col;
    wmma::store_matrix_sync(ptr_c, c_frag, ldc, wmma::mem_row_major);
}


__forceinline__ unsigned int div_and_ceil(float x, float y) {
    return ceil(x / y);
}


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_ARG(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_COMPATIBILITY(x, y, d1, d2) \
    TORCH_CHECK(x.size(d1) == y.size(d2), \
    #x " must be the same size by dim(" #d1 ") as " #y " by dim(" #d2 ")")

#define CHECK_SIZE_FOR_WMMA(x, wmma_size) TORCH_CHECK( \
    x.size(0) % wmma_size == 0.0 and x.size(1) % wmma_size == 0.0, \
    #x " size by all dimensions must be multiples of " #wmma_size " for WMMMA")


torch::Tensor matmul_tensor_cores_t(
    torch::Tensor matrix_a,
    torch::Tensor matrix_b) {

    const int WMMA_SIZE = 16;

    CHECK_CUDA_ARG(matrix_a);
    CHECK_CUDA_ARG(matrix_b);
    CHECK_COMPATIBILITY(matrix_a, matrix_b, 1, 1);
    CHECK_SIZE_FOR_WMMA(matrix_a, WMMA_SIZE);
    CHECK_SIZE_FOR_WMMA(matrix_b, WMMA_SIZE);

    int M = matrix_a.size(0);
    int N = matrix_b.size(0);
    int K = matrix_a.size(1);

    auto matrix_c = torch::zeros(
        {M, N},
        matrix_a.options().dtype(torch::kFloat32)
    );

    const int warp_size = 32;

    const dim3 block_size = {
        128, 4
    };

    dim3 grid_size = {
        div_and_ceil((warp_size * N) / WMMA_SIZE, block_size.x),
        div_and_ceil(M / WMMA_SIZE, block_size.y)
    };
    
    kernel_matmul_t_wmma<WMMA_SIZE, WMMA_SIZE, WMMA_SIZE><<<grid_size, block_size>>>(
        reinterpret_cast<half*>(matrix_a.data_ptr<c10::Half>()),
        reinterpret_cast<half*>(matrix_b.data_ptr<c10::Half>()),
        matrix_c.data_ptr<float>(),
        M, N, K
    );

    return matrix_c.toType(torch::kFloat16);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "matmul_tensor_cores_t",
        &matmul_tensor_cores_t,
        "Matrix multiplication with CUDA Tensor Cores (B matrix is transposed)"
    );
}