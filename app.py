from flask import Flask, make_response, render_template, Markup, request
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from predict import run, prepare_compact_dataset
import plotly.figure_factory as ff
import plotly.express as px
from plotly.offline import plot
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


START_DATE_TIME = pd.to_datetime('2021-02-01 14:00')

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


def plot_density(hour=15):

    df = pd.read_pickle("densities.p")

    df_section = df[df["Hour"] == hour] / 3600

    # fig = px.histogram(df_section, x="Means", histnorm='probability density')
    # fig.show()
    group_labels = ['distplot']
    fig = ff.create_distplot([df_section["Means"].values], group_labels, bin_size=len(df_section) / 20, curve_type="kde")
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        autosize=True,
        width=600,
        height=300,
        showlegend=False,
        template='plotly_white',
        xaxis_title="Session time in hours",
        yaxis_title="Probability",
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
def predict(timestamp_start=START_DATE_TIME, timestamp_end=START_DATE_TIME, forecast=False):

    if forecast:
        number_of_EVs, number_of_EVs_end = run(timestamp_start, timestamp_end)
        number_of_EVs, number_of_EVs_end = round(number_of_EVs), round(number_of_EVs_end)
    else:
        number_of_EVs, number_of_EVs_end = [], []
    
    print(number_of_EVs, number_of_EVs_end)
    div = main_plot()
    density_plot = plot_density(hour=timestamp_start.hour)
    observed_plot = observed_data()
    return render_template('home.html', 
                            start_date=timestamp_start.strftime("%Y-%m-%d"),
                            end_date=timestamp_end.strftime("%Y-%m-%d"),
                            start_time=str(timestamp_start.hour) + ":00",
                            end_time=str(timestamp_end.hour) + ":00",
                            selected_time=timestamp_start,
                            number_of_EVs=number_of_EVs, 
                            number_of_EVs_end=number_of_EVs_end,
                            plot=div, 
                            observed_plot=observed_plot,
                            density_plot=density_plot)


@server.route('/forecast/', methods=["POST", "GET"])
def forecast():
    # if request.method == 'GET':
    print(request.form)
    start_date, start_time = request.form.get("start_date_select"), request.form.get("start_time_select")
    end_date, end_time = request.form.get("end_date_select"), request.form.get("end_time_select")
    
    # error getting data
    if not start_date:
        return predict()

    start_str = start_date + " " + start_time + ':00'
    end_str = end_date + " " + end_time + ':00'
    print(start_str, end_str)
    timestamp_start = pd.to_datetime(start_str)
    timestamp_end = pd.to_datetime(end_str)
    return predict(timestamp_start=timestamp_start, timestamp_end=timestamp_end, forecast=True)

if __name__ == '__main__':
    server.run(debug=True, port=8080)


