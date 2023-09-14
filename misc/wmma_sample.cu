#include <mma.h>
#include <torch/extension.h>

using namespace nvcuda;


template <int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void kernel_matmul_wmma(half *a, half *b, float *c, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    int warp_x = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warp_y = (blockIdx.y * blockDim.y + threadIdx.y);

    int a_row = warp_y * WMMA_M;
    int b_col = warp_x * WMMA_N;

    if (a_row >= M or b_col >= N) {
        return;
    }

    wmma::fill_fragment(c_frag, 0.0f);

    int lda = K;
    int ldb = N;
    int ldc = N;

    for (int k = 0; k < K; k += WMMA_K) {
        int a_col = k;
        int b_row = k;

        half* ptr_a = a + lda * a_row + a_col;
        half* ptr_b = b + ldb * b_row + b_col;

        wmma::load_matrix_sync(a_frag, ptr_a, lda);
        wmma::load_matrix_sync(b_frag, ptr_b, ldb);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* ptr_c = c + ldc * warp_y * WMMA_M + warp_x * WMMA_N;
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
    x.dim(0) % wmma_size == 0.0 and x.dim(1) % wmma_size == 0.0, \
    #x " size by all dimensions must be multiples of " #wmma_size " for WMMMA")


torch::Tensor matmul_tensor_cores(
    torch::Tensor matrix_a,
    torch::Tensor matrix_b) {

    const int WMMA_SIZE = 16;

    CHECK_CUDA_ARG(matrix_a);
    CHECK_CUDA_ARG(matrix_b);
    CHECK_COMPATIBILITY(matrix_a, matrix_b, 1, 0);
    CHECK_SIZE_FOR_WMMA(matrix_a, WMMA_SIZE);
    CHECK_SIZE_FOR_WMMA(matrix_b, WMMA_SIZE);

    int N = matrix_a.size(0);
    int M = matrix_b.size(1);
    int K = matrix_b.size(0);

    auto matrix_c = torch::zeros(
        {matrix_a.size(0), matrix_b.size(1)},
        matrix_a.options().dtype(torch::kFloat32)
    );

    const dim3 block_size = {
        128, 4
    };

    dim3 grid_size = {
        div_and_ceil((warpSize * M) / WMMA_SIZE, block_size.x),
        div_and_ceil((warpSize * N) / WMMA_SIZE, block_size.y)
    };
    
    kernel_matmul_wmma<WMMA_SIZE, WMMA_SIZE, WMMA_SIZE><<<grid_size, block_size>>>(
        reinterpret_cast<half*>(matrix_a.data_ptr<c10::Half>()),
        reinterpret_cast<half*>(matrix_b.data_ptr<c10::Half>()),
        matrix_c.data_ptr<float>(),
        N, M, K
    );

    return matrix_c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "matmul_tensor_cores",
        &matmul_tensor_cores,
        "Matrix multiplication with CUDA Tensor Cores"
    );
}