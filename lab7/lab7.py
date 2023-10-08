import torch

from ..lab3.lab3 import (
    GenericTestCase,
    TestCaseFactory,
    Lab3TestCaseGrid2d,
)

from ..lab6.lab6 import (
    run_test_with_profiler,
    display_profile,
    Lab6TestCaseGrid2d
)


class Lab7TestCase(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float16],
    backward=True, backend='hs/lab7/lab7.cu', layout_x16=True
):

    pass


if __name__ == '__main__':
    prof = run_test_with_profiler(
        Lab3TestCaseGrid2d,
        Lab6TestCaseGrid2d,
        Lab7TestCase
    )

    display_profile(prof)
