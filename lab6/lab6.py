from ..lab3.lab3 import GenericTestCase, TestCaseFactory, Lab3TestCase
from torch.profiler import profile, ProfilerActivity
import unittest
import torch
import pandas as pd
import re
from io import StringIO


def run_test_with_profiler(
        *test_cases, activities=[ProfilerActivity.CUDA], verbosity=2):

    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
        for test_case in test_cases
    ])

    with profile(activities=activities) as prof:
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


class Lab6TestCase(
    GenericTestCase, metaclass=TestCaseFactory,
    dtypes=[torch.float64, torch.float32],
    backward=True, backend='hs/lab6/lab6.cu', wmma=False
):

    pass


if __name__ == '__main__':
    prof = run_test_with_profiler(Lab3TestCase, Lab6TestCase)

    df = parse_profile(
        prof, sort_by="cuda_time_total",
        row_limit=12
    )

    print(df)

    prof.export_chrome_trace('prof.out')
