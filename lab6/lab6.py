from ..lab3.lab3 import LinearFunction, LabTest as Lab3Test
from torch.profiler import profile, ProfilerActivity
import unittest


class LabTest(Lab3Test):
    @classmethod
    def setUpClass(cls):
        LinearFunction.up_backend('hs/lab6/lab6.cu')


def profile_test_case(test_case, row_limit):
    test_set = [
        getattr(test_case, attr_name)
        for attr_name in dir(test_case) if 'test_' in attr_name
    ]

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for test in test_set:
            try:
                test()
            except unittest.SkipTest:
                pass

    print(
        prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=row_limit)
    )


if __name__ == '__main__':
    Lab3Test.setUpClass()
    profile_test_case(Lab3Test(), row_limit=4)

    LabTest.setUpClass()
    profile_test_case(LabTest(), row_limit=4)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LabTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
