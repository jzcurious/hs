from ..lab3.lab3 import LinearFunction
import torch
from torch import nn
import math
import unittest


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        opt = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **opt))
        self.bias = nn.Parameter(torch.empty(out_features, **opt))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input) -> torch.Tensor:
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias=True'.format(
            self.in_features, self.out_features
        )


class LabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        LinearFunction.up_backend()

    def generic_case(self, dtype):
        opt = {'device': 'cuda', 'dtype': dtype}
        x = torch.ones(64, 9216, **opt)

        net1 = nn.Sequential(
            nn.Linear(9216, 4096, **opt),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096, **opt),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10, **opt),
            nn.Softmax(dim=1),
        )

        net2 = nn.Sequential(
            Linear(9216, 4096, **opt),
            nn.ReLU(inplace=True),
            Linear(4096, 4096, **opt),
            nn.ReLU(inplace=True),
            Linear(4096, 10, **opt),
            nn.Softmax(dim=1),
        )

        state_dict = net1.state_dict().copy()
        net2.load_state_dict(state_dict)

        y1 = net1(x)
        y2 = net2(x)

        match dtype:
            case torch.float16:
                tol = {'atol': 1e-3, 'rtol': 1e-2}
            case _:
                tol = {'atol': 1e-5, 'rtol': 1e-5}

        self.assertTrue(torch.allclose(y1, y2, **tol))

        y1.backward(torch.ones_like(y1))
        y2.backward(torch.ones_like(y2))

        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertTrue(torch.allclose(p1.grad, p2.grad, **tol))

    def test_float32(self):
        self.generic_case(torch.float32)

    @unittest.skipIf(torch.cuda.get_device_capability()[0] < 7,
                     'Unsupported CUDA device.')
    def test_float64(self):
        self.generic_case(torch.float64)


if __name__ == '__main__':
    unittest.main()
