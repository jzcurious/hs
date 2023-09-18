from ..lab3.lab3 import GenericTestCase, TestCaseFactory, Lab3TestCase
from torch.profiler import profile, ProfilerActivity
import unittest
import torch


def run_test_with_profiler(
        *test_cases, activities=[ProfilerActivity.CUDA],
        verbosity=2, row_limit=1):

    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
        for test_case in test_cases
    ])

    with profile(activities=activities) as prof:
        unittest.TextTestRunner(verbosity=verbosity).run(suite)

    print(
        prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=row_limit)
    )


class Lab6TestCase(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float64, torch.float32, torch.float16],
    verif=True, backward=True, backend='hs/lab6/lab6.cu', wmma=False
):

    pass


if __name__ == '__main__':
    run_test_with_profiler(Lab3TestCase, Lab6TestCase, row_limit=12)
