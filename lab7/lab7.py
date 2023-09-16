from ..lab3.lab3 import LinearFunction
from ..lab4.lab4 import LabTest as Lab4Test
from ..lab6.lab6 import profile_test_case
import unittest
import torch


class LabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        LinearFunction.up_backend('hs/lab7/lab7.cu')

    def test_float16(self):
        Lab4Test.test_float16(torch.float16)


if __name__ == '__main__':
    Lab4Test.setUpClass()
    profile_test_case(Lab4Test())

    LabTest.setUpClass()
    profile_test_case(LabTest())

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LabTest)
    unittest.TextTestRunner().run(suite)
