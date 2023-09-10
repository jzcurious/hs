import torch
from torch.utils.cpp_extension import load
import torch.utils.benchmark as benchmark
from tqdm import tqdm
import argparse
import pandas as pd
import plotly.graph_objects as go


my_ext = load(
    name='my_extension',
    sources=['lab2.cu'],
    extra_cuda_cflags=['-O3'],
    extra_cflags=['-O3'],
)


def binop_benchmark(low, high, step, binop_impl, dtype=torch.float32,
                    device='cuda:0', bar_label='', r=101):

    assert low < high
    assert step < high - low
    high += 1

    binop_benchmark.binop_impl_m = lambda a, b, m: binop_impl(a[:m], b[:m])

    if not hasattr(binop_benchmark, 'a'):
        binop_benchmark.a = torch.empty(
            (high, ), device=device, dtype=dtype).uniform_(-r, r)

        binop_benchmark.b = torch.empty(
            (high, ), device=device, dtype=dtype).uniform_(-r, r)

    stamps = {}
    num_test = (high - low) // step

    if not bar_label:
        bar_label = 'Vector Addition Benchmark'

    total_time = 0.0

    with tqdm(range(low, high, step), desc=bar_label, total=num_test) as test:
        for m in test:
            t = benchmark.Timer(
                    stmt='binop_benchmark.binop_impl_m(a, b, m)',
                    setup='from __main__ import binop_benchmark',
                    globals={
                        'a': binop_benchmark.a,
                        'b': binop_benchmark.b,
                        'm': m
                    }
                )

            timing = t.blocked_autorange().median
            stamps[m] = timing
            total_time += timing

            test.set_postfix(timing=f'{total_time:.3e}s')
            torch.cuda.synchronize()

    return stamps


def make_plot(df):
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df['Args size'], y=df['Time (ms)'],
                mode='lines+markers',
                line=dict(dash='solid', color='indianred'),
                name=''
            ),
        ],

        layout=go.Layout(
            xaxis=dict(
                range=[df["Args size"].min(), df["Args size"].max()],
                autorange=True,
                title_text="Args size"
            ),
            yaxis=dict(autorange=True, title_text="Time, ms"),
            title_text="Real complexity of binary operation",
            hovermode="closest",
            title_x=0.5,
            template="plotly_dark"),
    )

    fig.show()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('-a', '--low', type=int, default=51200)
    args_parser.add_argument('-b', '--high', type=int, default=5120000)
    args_parser.add_argument('-s', '--step', type=int, default=51200)
    args_parser.add_argument('-o', '--fname', type=str, default='')
    args_parser.add_argument('-p', '--plot', action='store_true')
    args = args_parser.parse_args()

    stamps = binop_benchmark(args.low, args.high, args.step, my_ext.my_add)
    df = pd.DataFrame({
        'Args size': list(stamps.keys()),
        'Time (ms)': list(stamps.values())
    })

    if args.plot:
        make_plot(df)

    if args.fname:
        df.to_csv(args.fname)
