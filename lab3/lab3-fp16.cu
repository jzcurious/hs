#include <torch/extension.h>


template <typename scalar_t>
__global__ void linear_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> weights,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> bias,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output) {

    auto k = blockIdx.x * blockDim.x + threadIdx.x;
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.z * blockDim.z + threadIdx.z;

    bool guard = i < weights.size(0) and j < weights.size(1) and k < input.size(0);

    if (guard) {
        auto part = input[k][i] * weights[i][j];

        if (i == 0) {
           part += bias[j];
        }

        if constexpr (std::is_same_v<scalar_t, c10::Half>) {
            atomicAdd(reinterpret_cast<__half*>(&output[k][j]), part);
        }
        else {
            atomicAdd(&output[k][j], part);
        }
    }
}


template <typename scalar_t>
__global__ void linear_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> weights,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> bias,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_weights,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_bias) {

    auto k = blockIdx.x * blockDim.x + threadIdx.x;
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.z * blockDim.z + threadIdx.z;

    bool guard = i < weights.size(0) and j < weights.size(1) and k < input.size(0);

    if (guard) {
        if constexpr (std::is_same_v<scalar_t, c10::Half>) {
            atomicAdd(reinterpret_cast<__half*>(&d_input[k][i]), weights[i][j]);
            atomicAdd(reinterpret_cast<__half*>(&d_weights[i][j]), input[k][i]);
        }
        else {
            atomicAdd(&d_input[k][i], weights[i][j]);
            atomicAdd(&d_weights[i][j], input[k][i]);
        }
    }

    if (i == 0 and k == 0) {
        d_bias[j] = input.size(0);
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


__forceinline__ std::tuple<dim3, dim3> configure_grid(
    unsigned int nx, unsigned int ny, unsigned int nz) {

    const dim3 block_size = {4, 8, 4};

    const dim3 grid_size = {
        div_and_ceil(nx, block_size.x),
        div_and_ceil(ny, block_size.y),
        div_and_ceil(nz, block_size.z)
    };

    return {grid_size, block_size}; 
}


torch::Tensor linear_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias) {

    CHECK_ARG(input);
    CHECK_ARG(weights);
    CHECK_ARG(bias);

    CHECK_COMPATIBILITY(input, weights, 1, 0);
    CHECK_COMPATIBILITY(bias, weights, 0, 1);

    auto output = torch::zeros({input.size(0), weights.size(1)}, input.options());

    dim3 grid_size, block_size;
    std::tie(grid_size, block_size) = configure_grid(
        input.size(0), input.size(1), weights.size(1));
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.type(),
        "linear_forward",
        ([&] {
            linear_forward_kernel<<<grid_size, block_size>>>(
                input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
            );
        })
    );

    return output;
}


std::vector<torch::Tensor> linear_backward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias) {

    CHECK_ARG(input);
    CHECK_ARG(weights);
    CHECK_ARG(bias);

    CHECK_COMPATIBILITY(input, weights, 1, 0);
    CHECK_COMPATIBILITY(bias, weights, 0, 1);

    auto d_input = torch::zeros_like(input);
    auto d_weights = torch::zeros_like(weights);
    auto d_bias = torch::zeros_like(bias);

    dim3 grid_size, block_size;
    std::tie(grid_size, block_size) = configure_grid(
        input.size(0), input.size(1), weights.size(1));
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.type(),
        "linear_backward",
        ([&] {
            linear_backward_kernel<<<grid_size, block_size>>>(
                input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                d_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                d_weights.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                d_bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
            );
        })
    );

    return {d_input, d_weights, d_bias};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "Custom linear layer (forward)");
    m.def("linear_backward", &linear_backward, "Custom linear layer (backward)");
}