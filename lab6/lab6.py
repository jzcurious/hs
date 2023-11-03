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
    # TODO: + parse numeric values

    sort_df = False

    try:
        s = prof.key_averages().table(**kwargs)
    except AttributeError:
        s = prof.key_averages().table()
        if 'sort_by' in kwargs:
            sort_df = True

    s = re.sub(r'-', '', s)
    s = str.join('\n', s.split('\n')[1:-4])

    s = re.sub(r'(?<=[\S])\s{1}(?=[\S])', '*', s)
    s = re.sub(r'(?<=[\S])\s{1}(?!\s*(\n|$))', ';', s)

    s = re.sub(r' ', '', s)
    s = re.sub(r'\*', ' ', s)

    df = pd.read_csv(StringIO(s), delimiter=';')

    if sort_df:
        df.sort_values(by=[kwargs['sort_by']], inplace=True)
        df.reset_index(drop=True, inplace=True)

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
    dtypes=[torch.float64, torch.float32, torch.float16],
    backward=True, backend='hs/lab6/lab6g2d.cu', layout_x16=True
):

    pass


class Lab6TestCaseGrid2dGuGoodLayout(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float64, torch.float32, torch.float16],
    backward=True, backend='hs/lab6/lab6g2d_gu.cu', layout_x16=True
):

    pass


class Lab6TestCaseGrid2dGuBadLayout(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float64, torch.float32, torch.float16],
    backward=True, backend='hs/lab6/lab6g2d_gu.cu', layout_x16=False
):

    pass


class Lab6TestCaseGrid3d(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float64, torch.float32, torch.float16],
    backward=True, backend='hs/lab6/lab6g3d.cu', layout_x16=True
):

    pass


def display_profile(prof, name_filter=r'linear.*float'):
    table_kwargs = {
        'sort_by': "cuda_time_total",
        'row_limit': 100
    }

    df = parse_profile(
        prof, **table_kwargs
    )

    df.drop(df.iloc[:, 1:6], inplace=True, axis=1)

    filtered_df = df[df.Name.str.contains(name_filter)]

    if is_notebook():
        from IPython.display import display
        display(filtered_df)
    else:
        print(end=2*'\n')
        print(filtered_df)

    prof.export_chrome_trace('prof.out')


if __name__ == '__main__':
    prof = run_test_with_profiler(
        Lab3TestCaseGrid2d,
        Lab3TestCaseGrid3d,
        Lab6TestCaseGrid2d,
        Lab6TestCaseGrid3d,
        Lab6TestCaseGrid2dGuGoodLayout,
        Lab6TestCaseGrid2dGuBadLayout
    )

    display_profile(prof)
