#include <torch/extension.h>


__global__ void d_add(float *a, float *b, float *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_SIZE(x, y) TORCH_CHECK(x.is_same_size(y), #y " must be the same size as " #x)
#define BLOCK_SIZE 128
#define CALC_GRID_SIZE(m) ((m + BLOCK_SIZE - 1) / BLOCK_SIZE)


torch::Tensor my_add(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_SIZE(a, b);

    auto c = torch::empty_like(a);
    int n = a.numel();

    d_add<<<CALC_GRID_SIZE(n), BLOCK_SIZE>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );

    return c;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_add", &my_add, "Custom vector addition");
}