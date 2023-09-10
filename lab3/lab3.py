import torch
from torch.utils.cpp_extension import load


class LinearFunction(torch.autograd.Function):
    ext = load(
        name='my_extension',
        sources=['hs/lab3/lab3.cu'],
        extra_cuda_cflags=[
            '-std=c++17',
            '--extended-lambda',
            '-O3'
        ],
        extra_cflags=['-O3'],
    )

    @staticmethod
    def forward(ctx, input, weights, bias):
        ctx.save_for_backward(input, weights, bias)
        return LinearFunction.ext.linear_forward(input, weights, bias)

    @staticmethod
    def backward(ctx, d_output):
        d_input, d_weights, d_bias = LinearFunction.ext.linear_backward(
            *ctx.saved_tensors, d_output)
        return d_input, d_weights, d_bias


def test(dtype=torch.float32):
    x = torch.rand((256, 1024), device='cuda', dtype=dtype, requires_grad=True)
    w1 = torch.rand((1024, 17), device='cuda', dtype=dtype, requires_grad=True)
    b1 = torch.rand((17, ), device='cuda', dtype=dtype, requires_grad=True)
    w2 = torch.rand((17, 8), device='cuda', dtype=dtype, requires_grad=True)
    b2 = torch.rand((8, ), device='cuda', dtype=dtype, requires_grad=True)

    y = LinearFunction.apply(x, w1, b1)
    z = LinearFunction.apply(y, w2, b2)

    z.backward(torch.ones_like(z))

    x_ = x.detach().clone().requires_grad_()
    w1_ = w1.detach().clone().requires_grad_()
    b1_ = b1.detach().clone().requires_grad_()
    w2_ = w2.detach().clone().requires_grad_()
    b2_ = b2.detach().clone().requires_grad_()

    y_ = x_ @ w1_ + b1_
    z_ = y_ @ w2_ + b2_

    z_.backward(torch.ones_like(z_))

    match dtype:
        case torch.float16 | torch.half:
            atol = 1e-3
            rtol = 1e-2
        case _:
            atol = 1e-5
            rtol = 1e-4

    assert torch.allclose(z_, z, atol=atol, rtol=rtol)
    assert torch.allclose(x_.grad, x.grad, atol=atol, rtol=rtol)
    assert torch.allclose(w1_.grad, w1.grad, atol=atol, rtol=rtol)
    assert torch.allclose(b1_.grad, b1.grad, atol=atol, rtol=rtol)
    assert torch.allclose(w2_.grad, w2.grad, atol=atol, rtol=rtol)
    assert torch.allclose(b2_.grad, b2.grad, atol=atol, rtol=rtol)

    print(
        "The test passed successfully.",
        f"[dtype={dtype}, atol={atol:.0e}, rtol={rtol:.0e}]"
    )


if __name__ == '__main__':
    torch.manual_seed(27)
    test(torch.float32)

    if torch.cuda.get_device_capability()[0] >= 7:
        test(torch.float16)
        test(torch.float64)
