from ..lab3.lab3 import GenericTestCase, TestCaseFactory, Lab3TestCase
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


class Lab6TestCase(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float64, torch.float32],
    backward=True, backend='hs/lab6/lab6.cu', wmma=False
):

    pass


if __name__ == '__main__':
    prof = run_test_with_profiler(Lab3TestCase, Lab6TestCase)
    # prof = run_test_with_profiler(Lab6TestCase)

    table_kwargs = {
        'sort_by': "cuda_time_total",
        'row_limit': 12
    }

    df = parse_profile(
        prof, **table_kwargs
    )

    filtered_df = df[df.Name.str.contains(r'linear.*float')]

    if is_notebook():
        from IPython.display import display
        display(filtered_df)
    else:
        print(end=2*'\n')
        print(filtered_df)

    prof.export_chrome_trace('prof.out')
