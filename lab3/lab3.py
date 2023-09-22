import torch
from torch.utils import cpp_extension
import unittest
from torch.nn.functional import linear as torch_linear


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
    @classmethod
    def setUpClass(cls):
        LinearFunction.up_backend(cls.backend)

    def _test_generic(self, dtype, verif, backward):
        if (dtype in [torch.float16, torch.float64]
                and torch.cuda.get_device_capability()[0] < 7):
            self.skipTest('Unsupported CUDA device.')

        tensor_opt = {
            'device': 'cuda',
            'dtype': dtype,
            'requires_grad': backward
        }

        match (dtype, verif):
            case (torch.float16, False):
                tol = {'atol': 1e-3, 'rtol': 1e-2}
            case (_, False):
                tol = {'atol': 1e-6, 'rtol': 1e-5}
            case (_, True):
                tol = {'atol': 1e-8, 'rtol': 1e-5}

        if verif:
            init_method = torch.ones
        else:
            init_method = torch.rand

        if self.wmma:
            x = init_method((128, 4096), **tensor_opt)
            w1 = init_method((2048, 4096), **tensor_opt)
            b1 = init_method((2048, ), **tensor_opt)
            w2 = init_method((16, 2048), **tensor_opt)
            b2 = init_method((16, ), **tensor_opt)
        else:
            x = init_method((127, 4097), **tensor_opt)
            w1 = init_method((2037, 4097), **tensor_opt)
            b1 = init_method((2037, ), **tensor_opt)
            w2 = init_method((15, 2037), **tensor_opt)
            b2 = init_method((15, ), **tensor_opt)

        y = LinearFunction.apply(x, w1, b1)
        z = LinearFunction.apply(y, w2, b2)

        x_ = x.detach().clone().requires_grad_()
        w1_ = w1.detach().clone().requires_grad_()
        b1_ = b1.detach().clone().requires_grad_()
        w2_ = w2.detach().clone().requires_grad_()
        b2_ = b2.detach().clone().requires_grad_()

        y_ = torch_linear(x_, w1_, b1_)
        z_ = torch_linear(y_, w2_, b2_)

        with torch.no_grad():
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
    def __add_test(attrs, backend, dtype, wmma, verif, backward):
        method_name = TestCaseFactory.__generate_test_name(
            backend, dtype, wmma, verif, backward
        )
        attrs[method_name] = \
            (lambda self, d=dtype, v=verif, b=backward:
                GenericTestCase._test_generic(self, d, v, b))

    @staticmethod
    def __add_tests(attrs, backend, dtypes, wmma, verif, backward):
        for dtype in dtypes:
            TestCaseFactory.__add_test(
                attrs, backend, dtype, wmma, False, backward)

            if verif:
                TestCaseFactory.__add_test(
                    attrs, backend, dtype, wmma, verif, backward)

    @staticmethod
    def __generate_test_name(backend, dtype, wmma, verif, backward):
        dtype_lb = str(dtype).split('.')[-1]
        wmma_lb = 'wmma' if wmma and dtype is torch.float16 else ''
        verif_lb = 'verif' if verif else 'prec'
        backend_lb = backend.split('/')[-1].replace('.', '_')
        bkwd_lb = 'forward_backward' if backward else 'forward'
        return f'test_{backend_lb}_{dtype_lb}_{wmma_lb}_{verif_lb}_{bkwd_lb}'


class Lab3TestCase(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float64, torch.float32],
    verif=True, backward=True, backend='hs/lab3/lab3.cu', wmma=False
):

    pass


if __name__ == '__main__':
    unittest.main()
