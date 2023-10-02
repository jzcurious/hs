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


template <typename scalar_t>
__global__ void linear_fwd_kern_g2d(
    const accessor_2d<scalar_t>input,
    const accessor_2d<scalar_t>weight,
    const accessor_1d<scalar_t>bias,
    accessor_2d<scalar_t> output) {

    auto n = blockIdx.x * blockDim.x + threadIdx.x;
    auto m = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= input.size(0) and n >=  weight.size(0)) {
        return;
    }

    scalar_t acc = 0;
    for (int k = 0; k < input.size(1); k++) {
        acc += input[m][k] * weight[n][k];
    }

    output[m][n] = acc + bias[n];
}


template <typename scalar_t>
__global__ void linear_bwd_kern_g2d(
    const accessor_2d<scalar_t> input,
    const accessor_2d<scalar_t> weight,
    const accessor_2d<scalar_t> d_output,
    accessor_2d<scalar_t> d_input,
    accessor_2d<scalar_t> d_weight,
    accessor_1d<scalar_t> d_bias) {

    auto n = blockIdx.x * blockDim.x + threadIdx.x;
    auto m = blockIdx.y * blockDim.y + threadIdx.y;

    /* dX = dY @ W */
    scalar_t acc = 0;
    if (m < d_output.size(0) and n < weight.size(1)) {
        for (int k = 0; k < d_output.size(1); k++) {
            acc += d_output[m][k] * weight[k][n];
        }
        d_input[m][n] = acc;
    }

    /* dW = dY^T @ X */
    acc = 0;
    if (m < d_output.size(1) and n < input.size(1)) {
        for (int k = 0; k < d_output.size(0); k++) {
            acc += d_output[k][m] * input[k][n];
        }
        d_weight[m][n] = acc;
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
            linear_fwd_kern_g2d<<<grid_dim, block_dim>>>(
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
            linear_bwd_kern_g2d<<<grid_dim, block_dim>>>(
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