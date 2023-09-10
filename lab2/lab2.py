import torch
from torch.utils.cpp_extension import load
from argparse import ArgumentParser


def test_add(add_impl, n=1024):
    x = torch.rand((n,), device='cuda')
    y = torch.rand((n,), device='cuda')
    z = add_impl(x, y)
    z_ = x + y

    assert torch.allclose(z, z_)

    print(f"The test passed successfully. [n = {n}]")


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        '-c', '--source', help='extension source code',
        default='lab2.cu', type=str
    )
    arg_parser.add_argument(
        '-n', '--worksize', help='length of vectors',
        default=10241, type=int
    )

    args = arg_parser.parse_args()

    my_ext = load(
        name='my_extension',
        sources=[args.source],
        extra_cuda_cflags=['-O3'],
        extra_cflags=['-O3'],
    )

    test_add(my_ext.my_add, n=args.worksize)
