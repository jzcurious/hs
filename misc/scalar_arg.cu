#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_ARG(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


template <typename scalar_t>
__global__ void broadcast_inplace_scalar_add_kernel_pass_by_ptr(
    scalar_t *tensor_data, const scalar_t *scalar_ptr, __uint32_t numel) {

    __uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numel) {
        tensor_data[i] += *scalar_ptr;
    }
}


__host__ torch::Tensor broadcast_inplace_scalar_add_pass_by_ptr(
    torch::Tensor tensor, torch::Tensor scalar_tensor) {

    CHECK_ARG(tensor);

    const dim3 block_dim = {128, 1, 1};
    const dim3 grid_dim = {ceil(tensor.numel() / block_dim.x), 1, 1};

    AT_DISPATCH_ALL_TYPES(
        tensor.scalar_type(),
        "linear_forward",
        ([&] {
            broadcast_inplace_scalar_add_kernel_pass_by_ptr<<<grid_dim, block_dim>>>(
                tensor.data_ptr<scalar_t>(),
                scalar_tensor.data_ptr<scalar_t>(),
                tensor.numel()
            );
        })
    );

    return tensor;
}


template <typename scalar_t>
__global__ void broadcast_inplace_scalar_add_kernel_pass_by_value(
    scalar_t *tensor_data, scalar_t scalar_value, __uint32_t numel) {

    __uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numel) {
        tensor_data[i] += scalar_value;
    }
}


__host__ torch::Tensor broadcast_inplace_scalar_add_pass_by_value(
    torch::Tensor tensor, torch::Tensor scalar_tensor) {

    CHECK_ARG(tensor);

    const dim3 block_dim = {128, 1, 1};
    const dim3 grid_dim = {ceil(tensor.numel() / block_dim.x), 1, 1};

    AT_DISPATCH_ALL_TYPES(
        tensor.scalar_type(),
        "linear_forward",
        ([&] {
            broadcast_inplace_scalar_add_kernel_pass_by_value<<<grid_dim, block_dim>>>(
                tensor.data_ptr<scalar_t>(),
                scalar_tensor[0].item<scalar_t>(),
                tensor.numel()
            );
        })
    );

    return tensor;
}


__host__ torch::Tensor broadcast_inplace_scalar_add_pass_by_value_cpptype(
    torch::Tensor tensor, float scalar) {

    CHECK_ARG(tensor);

    const dim3 block_dim = {128, 1, 1};
    const dim3 grid_dim = {ceil(tensor.numel() / block_dim.x), 1, 1};

    broadcast_inplace_scalar_add_kernel_pass_by_value<<<grid_dim, block_dim>>>(
        tensor.data_ptr<float>(),
        scalar,
        tensor.numel()
    );

    return tensor;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("broadcast_inplace_scalar_add_pass_by_ptr",
          &broadcast_inplace_scalar_add_pass_by_ptr,
          "Adds a scalar to each element of the vector (in place)"
    );
    m.def("broadcast_inplace_scalar_add_pass_by_value",
        &broadcast_inplace_scalar_add_pass_by_value,
        "Adds a scalar to each element of the vector (in place)"
    );
    m.def("broadcast_inplace_scalar_add_pass_by_value_cpptype",
        &broadcast_inplace_scalar_add_pass_by_value_cpptype,
        "Adds a scalar to each element of the vector (in place)"
    );
}