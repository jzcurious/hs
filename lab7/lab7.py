from ..lab3.lab3 import LinearFunction, LabTest as Lab3Test
from ..lab6.lab6 import profile_test_case
import unittest
import torch


class LabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        LinearFunction.up_backend('hs/lab7/lab7.cu')

    @unittest.skipIf(torch.cuda.get_device_capability()[0] < 7,
                     'Unsupported CUDA device.')
    def test_float16(self):
        super().generic_case(torch.float16, verif=False, use_layout_wmma=False)


if __name__ == '__main__':
    Lab3Test.setUpClass()
    profile_test_case(Lab3Test())

    LabTest.setUpClass()
    profile_test_case(LabTest())

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LabTest)
    unittest.TextTestRunner().run(suite)
