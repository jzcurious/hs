import torch
from torch.utils import cpp_extension
import unittest
import math
import torch.nn as nn
from torch.nn.functional import (
    linear as torch_linear,
    relu
)


class LinearFunction(torch.autograd.Function):
    r"""Evaluates the expression :math:`xA^T + b` and its gradient"""

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


class GenericTestCase(unittest.TestCase):
    @staticmethod
    def init_parameters(weight, bias):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)

    @classmethod
    def setUpClass(cls):
        LinearFunction.up_backend(cls.backend)

    def _test_generic(self, dtype, backward):
        if (dtype in [torch.float16, torch.float64]
                and torch.cuda.get_device_capability()[0] < 7):
            self.skipTest('Unsupported CUDA device.')

        tensor_opt = {
            'device': 'cuda',
            'dtype': dtype,
            'requires_grad': backward
        }

        match dtype:
            case torch.float16:
                tol = {'atol': 1e-3, 'rtol': 1e-2}
            case torch.float32:
                tol = {'atol': 1e-8, 'rtol': 1e-5}
            case torch.float64:
                tol = {'atol': 1e-16, 'rtol': 1e-10}

        if self.wmma:
            x = torch.rand((64, 9216), **tensor_opt)
            w1 = torch.empty((4096, 9216), **tensor_opt)
            b1 = torch.empty((4096, ), **tensor_opt)
            w2 = torch.empty((16, 4096), **tensor_opt)
            b2 = torch.empty((16, ), **tensor_opt)
        else:
            x = torch.rand((63, 9215), **tensor_opt)
            w1 = torch.empty((4095, 9215), **tensor_opt)
            b1 = torch.empty((4095, ), **tensor_opt)
            w2 = torch.empty((10, 4095), **tensor_opt)
            b2 = torch.empty((10, ), **tensor_opt)

        GenericTestCase.init_parameters(w1, b1)
        GenericTestCase.init_parameters(w2, b2)

        y = relu(LinearFunction.apply(x, w1, b1), inplace=True)
        z = relu(LinearFunction.apply(y, w2, b2), inplace=True)

        x_ = x.detach().clone().requires_grad_()
        w1_ = w1.detach().clone().requires_grad_()
        b1_ = b1.detach().clone().requires_grad_()
        w2_ = w2.detach().clone().requires_grad_()
        b2_ = b2.detach().clone().requires_grad_()

        y_ = relu(torch_linear(x_, w1_, b1_), inplace=True)
        z_ = relu(torch_linear(y_, w2_, b2_), inplace=True)

        with torch.no_grad():
            print((z - z_).abs().max().item())
            print(z_.abs().max().item())
            print(z.abs().max().item())
            self.assertTrue(torch.allclose(z_, z, **tol))

        if not backward:
            return

        z_.backward(torch.ones_like(z_))
        z.backward(torch.ones_like(z))

        self.assertTrue(torch.allclose(x_.grad, x.grad, **tol))
        self.assertTrue(torch.allclose(w1_.grad, w1.grad, **tol))
        self.assertTrue(torch.allclose(b1_.grad, b1.grad, **tol))
        self.assertTrue(torch.allclose(w2_.grad, w2.grad, **tol))
        self.assertTrue(torch.allclose(b2_.grad, b2.grad, **tol))


class TestCaseFactory(type):
    def __new__(cls, name, base, attrs, **kwargs):
        assert GenericTestCase in base
        attrs.update(kwargs)
        TestCaseFactory.__add_tests(attrs, **kwargs)
        return super().__new__(cls, name, base, attrs)

    @staticmethod
    def __add_test(attrs, backend, dtype, wmma, backward):
        method_name = TestCaseFactory.__generate_test_name(
            backend, dtype, wmma, backward
        )
        attrs[method_name] = \
            (lambda self, d=dtype, b=backward:
                GenericTestCase._test_generic(self, d, b))

    @staticmethod
    def __add_tests(attrs, backend, dtypes, wmma, backward):
        for dtype in dtypes:
            TestCaseFactory.__add_test(
                attrs, backend, dtype, wmma, backward)

    @staticmethod
    def __generate_test_name(backend, dtype, wmma, backward):
        dtype_lb = str(dtype).split('.')[-1]
        backend_lb = backend.split('/')[-1].replace('.', '_')
        bkwd_lb = 'forward_backward' if backward else 'forward'

        if wmma:
            return f'test_{backend_lb}_{dtype_lb}_wmma_{bkwd_lb}'
        else:
            return f'test_{backend_lb}_{dtype_lb}_{bkwd_lb}'


class Lab3TestCase(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float64, torch.float32],
    backward=True, backend='hs/lab3/lab3.cu', wmma=False
):

    pass


if __name__ == '__main__':
    unittest.main()
