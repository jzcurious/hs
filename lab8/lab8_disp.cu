#include <torch/extension.h>
#include "lab8_wmma.cu"
#include "lab8_smma.cu"
#include "lab8_g3d.cu"


bool maybe_wmma(torch::Tensor input, torch::Tensor weight) {
    if (input.scalar_type() != torch::kHalf) {
        return false;
    }

    if (input.size(0) % impl_wmma::wmma_dim.x != 0) {
        return false;
    }

    if (weight.size(0) % impl_wmma::wmma_dim.y != 0) {
        return false;
    }

    if (input.size(1) % impl_wmma::wmma_dim.z != 0) {
        return false;
    }

    return true;
}


bool maybe_smma(torch::Tensor input, torch::Tensor weight) {
    if (weight.size(0) % impl_smma::smma_dim.x != 0) {
        return false;
    }

    if (input.size(0) % impl_smma::smma_dim.y != 0) {
        return false;
    }
}


torch::Tensor linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    if (maybe_wmma(input, weight)) {
        return impl_wmma::linear_forward(input, weight, bias);
    }

    if (maybe_smma(input, weight)) {
        return impl_smma::linear_forward(input, weight, bias);
    }

    return impl_g3d::linear_forward(input, weight, bias);
}


std::vector<torch::Tensor> linear_backward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor d_output) {

    if (maybe_wmma(input, weight)) {
        return impl_wmma::linear_backward(input, weight, bias, d_output);
    }

    if (maybe_smma(input, weight)) {
        return impl_smma::linear_backward(input, weight, bias, d_output);
    }

    return impl_g3d::linear_backward(input, weight, bias, d_output);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_forward", &linear_forward, "Custom linear layer (forward)");
    m.def("linear_backward", &linear_backward, "Custom linear layer (backward)");
}