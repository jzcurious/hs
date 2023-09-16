import torch
from torch.utils import cpp_extension
import unittest


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def up_backend(backend_source='hs/lab3/lab3.cu'):
        LinearFunction.backend = cpp_extension.load(
            name='my_extension',
            sources=backend_source,
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
        return LinearFunction.backend.linear_forward(input, weights, bias)

    @staticmethod
    def backward(ctx, d_output):
        d_input, d_weights, d_bias = LinearFunction.backend.linear_backward(
            *ctx.saved_tensors, d_output)
        return d_input, d_weights, d_bias


class LabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        LinearFunction.up_backend()

    def generic_case(self, dtype):
        factory_kwargs = {
            'device': 'cuda',
            'dtype': dtype,
            'requires_grad': True
        }

        x = torch.rand((256, 1024), **factory_kwargs)
        w1 = torch.rand((1024, 17), **factory_kwargs)
        b1 = torch.rand((17, ), **factory_kwargs)
        w2 = torch.rand((17, 8), **factory_kwargs)
        b2 = torch.rand((8, ), **factory_kwargs)

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
                a = 1e-3
                r = 1e-2
            case _:
                a = 1e-5
                r = 1e-4

        self.assertTrue(torch.allclose(z_, z, atol=a, rtol=r))
        self.assertTrue(torch.allclose(x_.grad, x.grad, atol=a, rtol=r))
        self.assertTrue(torch.allclose(w1_.grad, w1.grad, atol=a, rtol=r))
        self.assertTrue(torch.allclose(b1_.grad, b1.grad, atol=a, rtol=r))
        self.assertTrue(torch.allclose(w2_.grad, w2.grad, atol=a, rtol=r))
        self.assertTrue(torch.allclose(b2_.grad, b2.grad, atol=a, rtol=r))

    def test_float32(self):
        self.generic_case(torch.float32)

    @unittest.skipIf(torch.cuda.get_device_capability()[0] < 7,
                     'Unsupported CUDA device.')
    def test_float16(self):
        self.generic_case(torch.float16)

    @unittest.skipIf(torch.cuda.get_device_capability()[0] < 7,
                     'Unsupported CUDA device.')
    def test_float64(self):
        self.generic_case(torch.float64)


if __name__ == '__main__':
    unittest.main()
