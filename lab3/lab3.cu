#include <torch/extension.h>


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
__global__ void linear_forward_kernel(
    const accessor_2d<scalar_t>input,
    const accessor_2d<scalar_t>weight,
    const accessor_1d<scalar_t>bias,
    accessor_2d<scalar_t> output) {

    auto k = blockIdx.x * blockDim.x + threadIdx.x;
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.z * blockDim.z + threadIdx.z;

    auto batch_size = input.size(0);
    auto weight_t_rows = weight.size(1);
    auto weight_t_cols = weight.size(0);

    bool guard = k < batch_size and i < weight_t_rows and j < weight_t_cols;

    if (guard) {
        auto part = input[k][i] * weight[j][i];

        if (i == 0) {
           part += bias[j];
        }

        gpuAtomicAdd(&output[k][j], part);
    }
}


template <typename scalar_t>
__global__ void linear_backward_kernel(
    const accessor_2d<scalar_t> input,
    const accessor_2d<scalar_t> weight,
    const accessor_2d<scalar_t> d_output,
    accessor_2d<scalar_t> d_input,
    accessor_2d<scalar_t> d_weight,
    accessor_1d<scalar_t> d_bias) {

    auto k = blockIdx.x * blockDim.x + threadIdx.x;
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.z * blockDim.z + threadIdx.z;

    auto batch_size = input.size(0);
    auto weight_t_rows = weight.size(1);
    auto weight_t_cols = weight.size(0);

    bool guard = k < batch_size and i < weight_t_rows and j < weight_t_cols;

    if (guard) {
        gpuAtomicAdd(&d_input[k][i], d_output[k][j] * weight[j][i]);
        gpuAtomicAdd(&d_weight[j][i], d_output[k][j] * input[k][i]);

        if (i == 0) {
            gpuAtomicAdd(&d_bias[j], d_output[k][j]);
        }
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
    torch::Tensor weight,
    torch::Tensor bias) {

    CHECK_ARG(input);
    CHECK_ARG(weight);
    CHECK_ARG(bias);

    CHECK_COMPATIBILITY(input, weight, 1, 1);
    CHECK_COMPATIBILITY(bias, weight, 0, 0);

    auto batch_size = input.size(0);
    auto weight_t_rows = weight.size(1);
    auto weight_t_cols = weight.size(0);

    auto output = torch::zeros(
        {batch_size, weight_t_cols},
        input.options()
    );

    dim3 grid_size, block_size;
    std::tie(grid_size, block_size) = configure_grid(
        batch_size, weight_t_rows, weight_t_cols);

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "linear_forward",
        ([&] {
            linear_forward_kernel<<<grid_size, block_size>>>(
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
    auto weight_t_rows = weight.size(1);
    auto weight_t_cols = weight.size(0);

    auto d_input = torch::zeros_like(input);
    auto d_weight = torch::zeros_like(weight);
    auto d_bias = torch::zeros_like(bias);

    dim3 grid_size, block_size;
    std::tie(grid_size, block_size) = configure_grid(
        batch_size, weight_t_rows, weight_t_rows);

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "linear_backward",
        ([&] {
            linear_backward_kernel<<<grid_size, block_size>>>(
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