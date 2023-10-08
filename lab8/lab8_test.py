import torch

from ..lab3.lab3 import (
    GenericTestCase,
    TestCaseFactory,
)

from ..lab6.lab6 import (
    run_test_with_profiler,
    display_profile,
)


class Lab8TestCase(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float16, torch.float32],
    backward=True, backend='hs/lab8/lab8_disp.cu', layout_x16=True
):

    pass


class Lab8TestCaseBadLayout(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float16, torch.float32],
    backward=True, backend='hs/lab8/lab8_disp.cu', layout_x16=False
):

    pass


if __name__ == '__main__':
    prof = run_test_with_profiler(
        Lab8TestCase, Lab8TestCaseBadLayout
    )

    display_profile(prof, name_filter=r'impl_')
