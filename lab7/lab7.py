from ..lab3.lab3 import GenericTestCase, TestCaseFactory
from ..lab6.lab6 import run_test_with_profiler
import torch


class Lab3TestCase(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float64, torch.float32],
    verif=True, backward=True, backend='hs/lab3/lab3.cu', wmma=True
):

    pass


class Lab7TestCase(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float64, torch.float32, torch.float16],
    verif=True, backward=True, backend='hs/lab7/lab7.cu', wmma=True
):

    pass


if __name__ == '__main__':
    run_test_with_profiler(Lab3TestCase, Lab7TestCase, row_limit=12)
