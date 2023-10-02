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


template <int batch_frag, int weight_cols_frag, int weight_rows_frag, typename scalar_t>
__global__ void linear_fwd_kern_smem(
    const accessor_2d<scalar_t> input,
    const accessor_2d<scalar_t> weight,
    const accessor_1d<scalar_t> bias,
    accessor_2d<scalar_t> output) {

    __shared__ scalar_t local_input[batch_frag][weight_cols_frag];          // 32: 128b .. 256b
    __shared__ scalar_t local_weight_t[weight_cols_frag][weight_rows_frag]; // 32: 128b .. 256b
    __shared__ scalar_t local_bias[weight_cols_frag];                       //  4: 16b  ..  32b
    __shared__ scalar_t local_output[batch_frag][weight_rows_frag];         // 16: 64b  .. 128b

    auto k = blockIdx.x * blockDim.x + threadIdx.x;
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.z * blockDim.z + threadIdx.z;

    auto l_k = threadIdx.x;
    auto l_i = threadIdx.y;
    auto l_j = threadIdx.z;

    auto batch_size = input.size(0);
    auto weight_rows = weight.size(0);
    auto weight_cols = weight.size(1);

    bool guard = k < batch_size and i < weight_cols and j < weight_rows;

    if (guard) {
        if (l_j == 0) {
            local_input[l_k][l_i] = input[k][i];
        }

        if (l_k == 0) {
            local_weight_t[l_i][l_j] = weight[j][i];
        }

        if (l_k == 0 and l_i == 0) {
            local_bias[l_j] = bias[j];
        }

        if (l_i == 0) {
            local_output[l_k][l_j] = 0;
        }
    }

    __syncthreads();

    if (guard) {
        auto part = local_input[l_k][l_i] * local_weight_t[l_i][l_j];

        if (i == 0) {
           part += local_bias[l_j];
        }
        
        gpuAtomicAdd(&local_output[l_k][l_j], part);
    }

    __syncthreads();

    if (guard and l_i == 0) {
        gpuAtomicAdd(&output[k][j], local_output[l_k][l_j]);
    }
}


template <int batch_frag, int weight_cols_frag, int weight_rows_frag, typename scalar_t>
__global__ void linear_bwd_kern_smem(
    const accessor_2d<scalar_t> input,
    const accessor_2d<scalar_t> weight,
    const accessor_2d<scalar_t> d_output,
    accessor_2d<scalar_t> d_input,
    accessor_2d<scalar_t> d_weight,
    accessor_1d<scalar_t> d_bias) {

    __shared__ scalar_t local_input[batch_frag][weight_cols_frag];            // 32: 128b .. 256b
    __shared__ scalar_t local_weight_t[weight_cols_frag][weight_rows_frag];   // 32: 128b .. 256b
    __shared__ scalar_t local_d_output[batch_frag][weight_rows_frag];         // 16:  64b .. 128b
    __shared__ scalar_t local_d_input[batch_frag][weight_cols_frag];          // 32: 128b .. 256b
    __shared__ scalar_t local_d_weight_t[weight_cols_frag][weight_rows_frag]; // 32: 128b .. 256b
    __shared__ scalar_t local_d_bias[weight_rows_frag];                       //  4:  16b ..  32b

    auto k = blockIdx.x * blockDim.x + threadIdx.x;
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.z * blockDim.z + threadIdx.z;

    auto l_k = threadIdx.x;
    auto l_i = threadIdx.y;
    auto l_j = threadIdx.z;

    auto batch_size = input.size(0);
    auto weight_rows = weight.size(0);
    auto weight_cols = weight.size(1);

    bool guard = k < batch_size and i < weight_cols and j < weight_rows;

    if (guard) {
        if (l_j == 0) {
            local_input[l_k][l_i] = input[k][i];
            local_d_input[l_k][l_i] = 0;
        }

        if (l_k == 0) {
            local_weight_t[l_i][l_j] = weight[j][i];
            local_d_weight_t[l_i][l_j] = 0;
        }
        
        if (l_i == 0) {
            local_d_output[l_k][l_j] = d_output[k][j];
        }

        if (l_k == 0 and l_i == 0) {
            local_d_bias[l_j] = 0;           
        }
    }

    __syncthreads();

    if (guard) {
        gpuAtomicAdd(&local_d_input[l_k][l_i],
            local_d_output[l_k][l_j] * local_weight_t[l_i][l_j]);
        
        gpuAtomicAdd(&local_d_weight_t[l_i][l_j],
            local_d_output[l_k][l_j] * local_input[l_k][l_i]);

        if (i == 0) {
            gpuAtomicAdd(&local_d_bias[l_j], local_d_output[l_k][l_j]);
        }
    }

    __syncthreads();

    if (guard) {
        if (l_j == 0) {
            gpuAtomicAdd(&d_input[k][i], local_d_input[l_k][l_i]);
        }

        if (l_k == 0) {
            gpuAtomicAdd(&d_weight[j][i], local_d_weight_t[l_i][l_j]);
        }

        if (l_k == 0 and l_i == 0) {
            gpuAtomicAdd(&d_bias[j], local_d_bias[l_j]);
        }
    }
}


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_ARG(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_COMPATIBILITY(x, y, d1, d2) \
    TORCH_CHECK_LINALG(x.size(d1) == y.size(d2), \
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
    auto weight_cols = weight.size(1);

    auto output = torch::zeros(
        {batch_size, weight_rows},
        input.options()
    );

    constexpr dim3 block_dim = {4, 8, 4};

    const dim3 grid_dim = {
        div_and_ceil(batch_size, block_dim.x),
        div_and_ceil(weight_cols, block_dim.y),
        div_and_ceil(weight_rows, block_dim.z)
    };

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "linear_forward",
        ([&] {
            linear_fwd_kern_smem<block_dim.x, block_dim.y, block_dim.z><<<grid_dim, block_dim>>>(
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

    constexpr dim3 block_dim = {4, 8, 4};

    const dim3 grid_dim = {
        div_and_ceil(batch_size, block_dim.x),
        div_and_ceil(weight_cols, block_dim.y),
        div_and_ceil(weight_rows, block_dim.z)
    };

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "linear_backward",
        ([&] {
            linear_bwd_kern_smem<block_dim.x, block_dim.y, block_dim.z><<<grid_dim, block_dim>>>(
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