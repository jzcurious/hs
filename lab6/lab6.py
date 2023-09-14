import torch
from ..lab3.lab3 import LinearFunction
from ..lab4.lab4 import test
from torch.utils import cpp_extension
from argparse import ArgumentParser
from torch.profiler import profile, ProfilerActivity


def load_backend(src):
    LinearFunction.up_backend(
        backend_impl=cpp_extension.load(
            name='my_extension',
            sources=[src],
            extra_cuda_cflags=[
                '-std=c++17',
                '--extended-lambda',
                '-O3'
            ],
            extra_cflags=['-O3'],
        )
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-s', '--source', help='backend source',
        type=str, default='hs/lab6/lab6.cu')
    args = parser.parse_args()

    load_backend(args.source)

    with profile(activities=[ProfilerActivity.CUDA], with_flops=True) as prof:
        test(torch.float32)
        if torch.cuda.get_device_capability()[0] >= 7:
            test(torch.float16)
            test(torch.float64)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
