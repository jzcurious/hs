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

    def generic_case(self, dtype, verif=False,
                     use_layout_wmma=False, backward=True):
        tensor_opt = {
            'device': 'cuda',
            'dtype': dtype,
            'requires_grad': True
        }

        match (dtype, verif):
            case (torch.float16, False):
                tol = {'atol': 1e-3, 'rtol': 1e-2}
            case (_, False):
                tol = {'atol': 1e-5, 'rtol': 1e-4}
            case (_, True):
                tol = {'atol': 1e-8, 'rtol': 1e-5}

        if verif:
            init_method = torch.ones
        else:
            init_method = torch.rand

        if use_layout_wmma:
            x = init_method((256, 1024), **tensor_opt)
            w1 = init_method((1024, 128), **tensor_opt)
            b1 = init_method((64, ), **tensor_opt)
            w2 = init_method((64, 16), **tensor_opt)
            b2 = init_method((16, ), **tensor_opt)
        else:
            x = init_method((257, 1023), **tensor_opt)
            w1 = init_method((1023, 132), **tensor_opt)
            b1 = init_method((132, ), **tensor_opt)
            w2 = init_method((132, 10), **tensor_opt)
            b2 = init_method((10, ), **tensor_opt)

        y = LinearFunction.apply(x, w1, b1)
        z = LinearFunction.apply(y, w2, b2)

        x_ = x.detach().clone().requires_grad_()
        w1_ = w1.detach().clone().requires_grad_()
        b1_ = b1.detach().clone().requires_grad_()
        w2_ = w2.detach().clone().requires_grad_()
        b2_ = b2.detach().clone().requires_grad_()

        y_ = x_ @ w1_ + b1_
        z_ = y_ @ w2_ + b2_

        with torch.no_grad():
            self.assertTrue(torch.allclose(z_, z, **tol))

        if not backward:
            return

        z.backward(torch.ones_like(z))
        z_.backward(torch.ones_like(z_))

        self.assertTrue(torch.allclose(x_.grad, x.grad, **tol))
        self.assertTrue(torch.allclose(w1_.grad, w1.grad, **tol))
        self.assertTrue(torch.allclose(b1_.grad, b1.grad, **tol))
        self.assertTrue(torch.allclose(w2_.grad, w2.grad, **tol))
        self.assertTrue(torch.allclose(b2_.grad, b2.grad, **tol))

    def test_verification_float32(self):
        self.generic_case(torch.float32, verif=True)

    @unittest.skipIf(torch.cuda.get_device_capability()[0] < 7,
                     'Unsupported CUDA device.')
    def test_verification_float64(self):
        self.generic_case(torch.float64, verif=True)

    def test_precision_float32(self):
        self.generic_case(torch.float32)

    @unittest.skipIf(torch.cuda.get_device_capability()[0] < 7,
                     'Unsupported CUDA device.')
    def test_precision_float64(self):
        self.generic_case(torch.float64)


if __name__ == '__main__':
    unittest.main()
