#include <torch/extension.h>
#include <mma.h>

using namespace nvcuda;

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


template <int wmma_m, int wmma_n, int wmma_k>
__global__ void linear_forward_kernel_wmma(
    const accessor_2d<c10::Half>input,
    const accessor_2d<c10::Half>weight,
    const accessor_1d<c10::Half>bias,
    accessor_2d<float> output) {

    wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, float> c_frag;

    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_x = thread_x / warpSize;
    int warp_y = (blockIdx.y * blockDim.y + threadIdx.y);

    int input_row = warp_y * wmma_m;
    int weight_col = warp_x * wmma_n;

    if (input_row >= input.size(0) or weight_col >= weight.size(1)) {
        return;
    }

    wmma::fill_fragment(c_frag, 0.0f);

    int lda = input.size(1);  // K
    int ldb = weight.size(1); // N
    int ldc = weight.size(1); // N

    for (int k = 0; k < lda; k += wmma_k) {
        int input_col = k;
        int weight_row = k;

        half* ptr_a = reinterpret_cast<half*>(input.data()) \
            + lda * input_row + input_col;
        
        half* ptr_b = reinterpret_cast<half*>(weight.data()) \
            + ldb * weight_row + weight_col;

        wmma::load_matrix_sync(a_frag, ptr_a, lda);
        wmma::load_matrix_sync(b_frag, ptr_b, ldb);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* ptr_c = output.data() \
        + ldc * warp_y * wmma_m + warp_x * wmma_n;

    wmma::store_matrix_sync(ptr_c, c_frag, ldc, wmma::mem_row_major);

    int warp_offset_x = warp_x * warpSize;
    int thread_x_tile = thread_x - warp_offset_x;

    if (thread_x_tile < wmma_n) {
        for (int i = 0; i < wmma_m; i++) {
            output[input_row + i][weight_col + thread_x_tile] \
                += static_cast<float>(bias[weight_col + thread_x_tile]);
        }
    }
}


template <typename scalar_t>
__global__ void linear_forward_kernel_no_wmma(
    const accessor_2d<scalar_t>input,
    const accessor_2d<scalar_t>weight,
    const accessor_1d<scalar_t>bias,
    accessor_2d<scalar_t> output) {

    auto k = blockIdx.x * blockDim.x + threadIdx.x;
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.z * blockDim.z + threadIdx.z;

    bool guard = i < weight.size(0) and j < weight.size(1) and k < input.size(0);

    if (guard) {
        auto part = input[k][i] * weight[i][j];

        if (i == 0) {
           part += bias[j];
        }

        gpuAtomicAdd(&output[k][j], part);
    }
}


template <int wmma_m, int wmma_n, int wmma_k>
__global__ void linear_backward_kernel_wmma(
    const accessor_2d<c10::Half> input,
    const accessor_2d<c10::Half> weight,
    const accessor_2d<c10::Half> d_output,
    accessor_2d<float> d_input,
    accessor_2d<float> d_weight,
    accessor_1d<float> d_bias) {

    wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, half, wmma::col_major> input_t_a_frag;
    wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, half, wmma::row_major> d_output_a_frag;
    wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, half, wmma::row_major> d_output_b_frag;
    wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, half, wmma::col_major> weight_t_b_frag;
    wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, float> d_input_c_frag;
    wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, float> d_weight_c_frag;

    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_x = thread_x / warpSize;
    int warp_y = (blockIdx.y * blockDim.y + threadIdx.y);

    int d_output_row = warp_y * wmma_m; // k
    int weight_t_col = warp_x * wmma_n; // i
    int input_t_row = weight_t_col;     // i

    if (d_output_row >= d_output.size(0) or weight_t_col >= weight.size(0)) {
        return;
    }

    wmma::fill_fragment(d_input_c_frag, 0.0f);
    wmma::fill_fragment(d_weight_c_frag, 0.0f);

    int lda = d_output.size(1);  // K
    int ldb = weight.size(1);    // N
    int ldc = weight.size(0);    // K

    for (int k = 0; k < lda; k += wmma_k) {
        int d_output_col = k;
        int weight_t_row = k;

        half* ptr_d_output = reinterpret_cast<half*>(d_output.data()) \
            + lda * d_output_row + d_output_col;
        
        half* ptr_weight_t = reinterpret_cast<half*>(weight.data()) \
            + ldb * weight_t_col + weight_t_row;

        wmma::load_matrix_sync(d_output_a_frag, ptr_d_output, lda);
        wmma::load_matrix_sync(weight_t_b_frag, ptr_weight_t, ldb);

        wmma::mma_sync(
            d_input_c_frag, d_output_a_frag, weight_t_b_frag, d_input_c_frag);
    }

    float* ptr_d_input = d_input.data() \
        + ldc * warp_y * wmma_m + warp_x * wmma_n;

    wmma::store_matrix_sync(ptr_d_input, d_input_c_frag, ldc, wmma::mem_row_major);
}


template <typename scalar_t>
__global__ void linear_backward_kernel_no_wmma(
    const accessor_2d<scalar_t> input,
    const accessor_2d<scalar_t> weight,
    const accessor_2d<scalar_t> d_output,
    accessor_2d<scalar_t> d_input,
    accessor_2d<scalar_t> d_weight,
    accessor_1d<scalar_t> d_bias) {

    auto k = blockIdx.x * blockDim.x + threadIdx.x;
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.z * blockDim.z + threadIdx.z;

    bool guard = i < weight.size(0) and j < weight.size(1) and k < input.size(0);

    if (guard) {
        gpuAtomicAdd(&d_input[k][i], d_output[k][j] * weight[i][j]);
        gpuAtomicAdd(&d_weight[i][j], d_output[k][j] * input[k][i]);

        if (i == 0) {
            gpuAtomicAdd(&d_bias[j], d_output[k][j]);
        }
    }
}


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_ARG(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_COMPATIBILITY(x, y, d1, d2) \
    TORCH_CHECK_LINALG(x.size(d1) == y.size(d2), \
    #x " must be the same size by dim(" #d1 ") as " #y " by dim(" #d2 ")")

#define CHECK_SIZE_FOR_WMMA_A_M(a, wmma_m) TORCH_CHECK_VALUE( \
    a.size(0) % wmma_m == 0.0, \
    #a " size by all dimensions must be multiples of " #wmma_m " for WMMMA")

#define CHECK_SIZE_FOR_WMMA_A_K(a, wmma_k) TORCH_CHECK_VALUE( \
    a.size(1) % wmma_k == 0.0, \
    #a " size by all dimensions must be multiples of " #wmma_k " for WMMMA")

#define CHECK_SIZE_FOR_WMMA_B_K(b, wmma_k) TORCH_CHECK_VALUE( \
    b.size(0) % wmma_k == 0.0, \
    #b " size by all dimensions must be multiples of " #wmma_k " for WMMMA")

#define CHECK_SIZE_FOR_WMMA_B_N(b, wmma_n) TORCH_CHECK_VALUE( \
    b.size(1) % wmma_n == 0.0, \
    #b " size by all dimensions must be multiples of " #wmma_n " for WMMMA")

#define CHECK_WMMA(a, b, wmma_dim) \
    CHECK_SIZE_FOR_WMMA_A_M(a, wmma_dim.x); CHECK_SIZE_FOR_WMMA_A_K(a, wmma_dim.z); \
    CHECK_SIZE_FOR_WMMA_B_K(b, wmma_dim.z); CHECK_SIZE_FOR_WMMA_B_N(b, wmma_dim.y)


__forceinline__ unsigned int div_and_ceil(float x, float y) {
    return ceil(x / y);
}


torch::Tensor linear_forward_wmma(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    CHECK_ARG(input);
    CHECK_ARG(weight);
    CHECK_ARG(bias);

    CHECK_COMPATIBILITY(input, weight, 1, 0);
    CHECK_COMPATIBILITY(bias, weight, 0, 1);

    constexpr dim3 wmma_dim = {
        16, // M
        16, // N
        16  // K
    };

    CHECK_WMMA(input, weight, wmma_dim);

    auto opt = input.options().dtype(torch::kFloat);
    auto output = torch::zeros({input.size(0), weight.size(1)}, opt);

    const int warp_size = 32;
    const dim3 block_dim = {128, 2};

    dim3 grid_dim = {
        div_and_ceil(weight.size(1), wmma_dim.x) * warp_size,
        div_and_ceil(weight.size(0), wmma_dim.y)
    };

    linear_forward_kernel_wmma<wmma_dim.x, wmma_dim.y, wmma_dim.z><<<grid_dim, block_dim>>>(
        input.packed_accessor32<c10::Half, 2, torch::RestrictPtrTraits>(),
        weight.packed_accessor32<c10::Half, 2, torch::RestrictPtrTraits>(),
        bias.packed_accessor32<c10::Half, 1, torch::RestrictPtrTraits>(),
        output.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    return output.toType(torch::kFloat16);
}


torch::Tensor linear_forward_no_wmma(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    CHECK_ARG(input);
    CHECK_ARG(weight);
    CHECK_ARG(bias);

    CHECK_COMPATIBILITY(input, weight, 1, 0);
    CHECK_COMPATIBILITY(bias, weight, 0, 1);

    auto output = torch::zeros({input.size(0), weight.size(1)}, input.options());

    constexpr dim3 block_dim = {4, 8, 4};

    const dim3 grid_dim = {
        div_and_ceil(input.size(0), block_dim.x),
        div_and_ceil(input.size(1), block_dim.y),
        div_and_ceil(weight.size(1), block_dim.z)
    };

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "linear_forward",
        ([&] {
            linear_forward_kernel_no_wmma<<<grid_dim, block_dim>>>(
                input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                weight.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
            );
        })
    );

    return output;
}


torch::Tensor linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    if (typeid(input.scalar_type()) == typeid(torch::kFloat16)) {
        try {
            return linear_forward_wmma(input, weight, bias);
        }
        catch (...) { // fix me
            return linear_forward_no_wmma(input, weight, bias);
        }
    }
    else {
        return linear_forward_no_wmma(input, weight, bias);
    }
}


std::vector<torch::Tensor> linear_backward_wmma(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor d_output) {

    CHECK_ARG(input);
    CHECK_ARG(weight);
    CHECK_ARG(bias);
    CHECK_ARG(d_output);

    CHECK_COMPATIBILITY(input, weight, 1, 0);
    CHECK_COMPATIBILITY(bias, weight, 0, 1);
    CHECK_COMPATIBILITY(d_output, weight, 1, 1);
    CHECK_COMPATIBILITY(d_output, bias, 1, 0);

    constexpr dim3 wmma_dim = {
        16, // M
        16, // N
        16  // K
    };

    CHECK_WMMA(input, weight, wmma_dim);

    auto opt = input.options().dtype(torch::kFloat);
    auto d_input = torch::zeros({input.size(0), input.size(1)}, opt);
    auto d_weight = torch::zeros({weight.size(0), weight.size(1)}, opt);
    auto d_bias = torch::zeros({bias.size(0)}, opt);

    const int warp_size = 32;
    const dim3 block_dim = {128, 2};

    dim3 grid_dim = {
        div_and_ceil(d_output.size(0), wmma_dim.x) * warp_size,
        div_and_ceil(weight.size(0), wmma_dim.y)
    };

    linear_backward_kernel_wmma<wmma_dim.x, wmma_dim.y, wmma_dim.z><<<grid_dim, block_dim>>>(
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


std::vector<torch::Tensor> linear_backward_no_wmma(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor d_output) {

    CHECK_ARG(input);
    CHECK_ARG(weight);
    CHECK_ARG(bias);
    CHECK_ARG(d_output);

    CHECK_COMPATIBILITY(input, weight, 1, 0);
    CHECK_COMPATIBILITY(bias, weight, 0, 1);
    CHECK_COMPATIBILITY(d_output, weight, 1, 1);
    CHECK_COMPATIBILITY(d_output, bias, 1, 0);

    auto d_input = torch::zeros_like(input);
    auto d_weight = torch::zeros_like(weight);
    auto d_bias = torch::zeros_like(bias);

    constexpr dim3 block_dim = {4, 8, 4};

    const dim3 grid_dim = {
        div_and_ceil(input.size(0), block_dim.x),
        div_and_ceil(input.size(1), block_dim.y),
        div_and_ceil(weight.size(1), block_dim.z)
    };

    AT_DISPATCH_FLOATING_TYPES(
        input.type(),
        "linear_backward",
        ([&] {
            linear_backward_kernel_no_wmma<<<grid_dim, block_dim>>>(
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


std::vector<torch::Tensor> linear_backward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor d_output) {

    if (typeid(input.scalar_type()) == typeid(torch::kFloat16)) {
        try {
            return linear_backward_wmma(input, weight, bias, d_output);
        }
        catch (...) { // fix me
            return linear_backward_no_wmma(input, weight, bias, d_output);
        }
    }
    else {
        return linear_backward_no_wmma(input, weight, bias, d_output);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "Custom linear layer (forward)");
    m.def("linear_backward", &linear_backward, "Custom linear layer (backward)");
}