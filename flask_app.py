from flask import Flask, jsonify, make_response, render_template, Markup
from dash import Dash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from predict import run, prepare_compact_dataset
import plotly
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objs as go
from bs4 import BeautifulSoup
from plotly.offline import plot
import pandas as pd
import numpy as np


server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname='/dash/'
)

section_df = pd.read_pickle("results.p")
graph = px.bar(x=section_df.index, y=section_df.Count)
graph.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                marker_line_width=1.5, opacity=0.6)
                
app.layout = html.Div(
    html.Div([
        html.H4('Charging EV count predictions'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph', figure=graph),
        dcc.Interval(
            id='predictions',
            interval=10000, # in milliseconds
            n_intervals=0
        )
    ])
)


@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    section_df = pd.read_pickle("results.p")
    graph = px.bar(x=section_df.index, y=section_df.Count)
    graph.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                    marker_line_width=1.5, opacity=0.6)

    return graph


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):

    # Create the graph with subplots

    section_df = pd.read_pickle("results.p")
    graph = px.bar(x=section_df.index, y=section_df.Count)
    graph.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                    marker_line_width=1.5, opacity=0.6)

    return graph


def plot_density():

    df = pd.read_pickle("densities.p")

    df_section = df[df["Hour"] == 15]

    # fig = px.histogram(df_section, x="Means", histnorm='probability density')
    # fig.show()
    group_labels = ['distplot']
    fig = ff.create_distplot([df_section["Means"].values], group_labels, bin_size=1000, curve_type="kde")
    fig.update_layout(
        margin=dict(l=30, r=0, t=0, b=30),
        autosize=True,
        width=600,
        height=300,
        showlegend=False,
        template='plotly_white',
    )
    return plot(fig, auto_open=False, output_type='div')


def observed_data():
    counted_df = prepare_compact_dataset()
    fig = px.bar(x=counted_df.index, y=counted_df.session_time)
    fig.update_layout(
        margin=dict(l=30, r=0, t=30, b=30),
        autosize=True,
        width=900,
        height=300,
        showlegend=False,
        template='plotly_white',
        xaxis_title="Date",
        yaxis_title="Charging cars per day",
    )
    fig.update_traces(marker_color='black')
    fig.add_vrect(x0=pd.to_datetime('2020-03-05'), x1=pd.to_datetime('2020-04-15'), 
                        line_width=0, fillcolor="rgb(158,202,225)", opacity=0.4)
    fig.add_annotation(x=pd.to_datetime('2020-03-05'), 
                        y=np.max(counted_df.session_time),
            text="Start pandemic",
            showarrow=False,
            yshift=10)
    return plot(fig, auto_open=False, output_type='div')


def main_plot():
    section_df = pd.read_pickle("results.p")
    graph = px.bar(x=section_df.index, y=section_df.Count)
    graph.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                    marker_line_width=1.5, opacity=0.6)
    graph.update_layout(
        autosize=True,
        margin=dict(l=30, r=0, t=0, b=30),
        width=900,
        template='plotly_white',
        xaxis_title="Date",
        yaxis_title="Charging cars per hour",
    )
    return plot(graph, auto_open=False, output_type='div')

@server.route('/')
def predict():

    # TIJDELIJK UITGECOMMENT WANT ANDERS DUURT HET TE LANG
    # number_of_EVs, number_of_EVs_end = run()
    # print(number_of_EVs, number_of_EVs_end)

    selected_time = pd.to_datetime("2021-04-04 14:00")

    number_of_EVs, number_of_EVs_end = 40, 5
 
    div = main_plot()
    density_plot = plot_density()
    observed_plot = observed_data()

    return render_template('home.html', 
                            selected_time=selected_time,
                            number_of_EVs=number_of_EVs, 
                            number_of_EVs_end=number_of_EVs_end,
                            plot=div, 
                            observed_plot=observed_plot,
                            density_plot=density_plot)


@server.route("/dash")
def my_dash_app():
    return app.index()



if __name__ == '__main__':
    server.run(debug=True, port=8080)


