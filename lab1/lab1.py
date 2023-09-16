import pandas as pd
import plotly.graph_objects as go
import argparse


def make_plot(df):
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["Vector Length"], y=df["CPU Time"],
                mode="lines+markers",
                line=dict(dash='solid', color='indianred'),
                name="CPU"
            ),
            go.Scatter(
                x=df["Vector Length"], y=df["GPU Time"],
                mode="lines+markers",
                line=dict(dash='solid', color='limegreen'),
                name="GPU"
            )
        ],

        layout=go.Layout(
            xaxis=dict(
                range=[df["Vector Length"].min(), df["Vector Length"].max()],
                autorange=True,
                title_text="Vector Length"
            ),
            yaxis=dict(autorange=True, title_text="Time, ms"),
            title_text="Real complexity of operation",
            hovermode="closest",
            title_x=0.5,
            template="plotly_dark"),
    )

    fig.show()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '--stmpfile', type=str, default='timings.stmp',
        help='Timings from lab1'
    )
    args = args_parser.parse_args()

    df = pd.read_csv(args.stmpfile, skipinitialspace=True)
    make_plot(df)
