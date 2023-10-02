from ..lab3.lab3 import (
    GenericTestCase,
    TestCaseFactory,
    Lab3TestCaseGrid2d,
    Lab3TestCaseGrid3d
)
from torch.profiler import profile, ProfilerActivity
import unittest
import torch
import pandas as pd
import re
from io import StringIO


def run_test_with_profiler(
        *test_cases,
        activities=[ProfilerActivity.CUDA],
        verbosity=2,
        **profiler_kwargs
):

    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
        for test_case in test_cases
    ])

    with profile(activities=activities, **profiler_kwargs) as prof:
        unittest.TextTestRunner(verbosity=verbosity).run(suite)

    return prof


def parse_profile(prof, **kwargs):
    s = prof.key_averages().table(**kwargs)

    s = re.sub(r'-', '', s)
    s = str.join('\n', s.split('\n')[1:-4])

    s = re.sub(r'(?<=[\S])\s{1}(?=[\S])', '*', s)
    s = re.sub(r'(?<=[\S])\s{1}(?!\s*(\n|$))', ';', s)

    s = re.sub(r' ', '', s)
    s = re.sub(r'\*', ' ', s)

    df = pd.read_csv(StringIO(s), delimiter=';')
    return df


def is_notebook():
    try:
        if get_ipython().__class__.__name__:
            return True
        else:
            return False
    except NameError:
        return False


class Lab6TestCaseGrid2d(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float64, torch.float32],
    backward=True, backend='hs/lab6/lab6g2d.cu', layout_x16=True
):

    pass


class Lab6TestCaseGrid3d(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float64, torch.float32],
    backward=True, backend='hs/lab6/lab6g3d.cu', layout_x16=True
):

    pass


if __name__ == '__main__':
    prof = run_test_with_profiler(
        Lab3TestCaseGrid2d,
        Lab3TestCaseGrid3d,
        Lab6TestCaseGrid2d,
        Lab6TestCaseGrid3d
    )
    # prof = run_test_with_profiler(Lab6TestCase)

    table_kwargs = {
        'sort_by': "cuda_time_total",
        'row_limit': 20
    }

    df = parse_profile(
        prof, **table_kwargs
    )

    df.drop(df.iloc[:, 1:6], inplace=True, axis=1)

    filtered_df = df[df.Name.str.contains(r'linear.*float')]

    if is_notebook():
        from IPython.display import display
        display(filtered_df)
    else:
        print(end=2*'\n')
        print(filtered_df)

    prof.export_chrome_trace('prof.out')
