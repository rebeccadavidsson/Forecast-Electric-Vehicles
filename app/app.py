from flask import Flask, make_response, render_template, Markup, request
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.figure_factory as ff
import plotly.express as px
from plotly.offline import plot
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import plotly.graph_objects as go
from collections import Counter
import datetime as dt
try:
    from .predict import run_model, prepare_compact_dataset
except ImportError:
    from predict import run_model, prepare_compact_dataset

START_DATE_TIME = pd.to_datetime('2021-02-01 14:00')

server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname='/dash/'
)


app.layout = html.Div(
    html.Div([
        html.H4('Charging EV count predictions'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph'),
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
    section_df = pd.read_pickle("app/static/results.p")
    graph = px.bar(x=section_df.index, y=section_df.Count)
    graph.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                    marker_line_width=1.5, opacity=0.6)

    return graph


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):

    # Create the graph with subplots

    section_df = pd.read_pickle("app/static/results.p")
    graph = px.bar(x=section_df.index, y=section_df.Count)
    graph.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                    marker_line_width=1.5, opacity=0.6)

    return graph


def plot_density(hour=15, day=None):

    df = pd.read_pickle("app/static/densities.p")

    if day:
        # print(day.day)
        new_day = day
        # get mean 
        session_means = pd.read_pickle("app/static/week_days.p")
        print(session_means[session_means.index == new_day]['normalize'])
        session_mean_val = session_means[session_means.index == new_day]['normalize'].values[0]
        print(session_mean_val)
        if session_mean_val > 1:
            session_mean_val = 1
    else:
        session_mean_val = 1

    df_section = df[df["Hour"] == hour] / 3600
    new_section = len(df_section[df_section["Means"] >= 4])
    try:
        probability = round((new_section / len(df_section) * 100 * session_mean_val), 2)
    except:
        probability = 0

    # fig = px.histogram(df_section, x="Means", histnorm='probability density')
    # fig.show()
    group_labels = ['distplot']
    fig = ff.create_distplot(
        [df_section["Means"].values], group_labels, show_hist=False, bin_size=len(df_section) / 20, curve_type="kde")
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

    scatter = px.scatter(x=df["Hour"], y=df["Means"], trendline="lowess")
    scatter.update_layout(
        margin=dict(l=50, r=40, t=40, b=60),
        autosize=True,
        width=900,
        height=350,
        showlegend=False,
        template='plotly_white',
        xaxis_title="Hour",
        yaxis_title="Session time",
    )
    scatter.update_traces(
        mode='markers',
        marker=dict(color='gold'),
    )
    return [plot(fig, auto_open=False, output_type='div'),
            plot(scatter, auto_open=False, output_type='div'), probability]


def sessions(time_start, time_end):
    start_df = pd.read_pickle("app/static/predictions_timestamp_start.p")
    end = pd.read_pickle("app/static/predictions_timestamp_end.p")
    merged_df = pd.merge(start_df, end, left_index=True, right_index=True)
    merged_df = merged_df.rename(columns={"0_y": "Charging", "0_x": "Ending" })
    merged_section = merged_df[(merged_df.index >= time_start.floor("D") + dt.timedelta(hours=1) ) & (merged_df.index <= time_end.ceil("D"))]
    merged_section = merged_section.reset_index()
    merged_section["New"] = abs(merged_section["Charging"].diff())
    merged_section["Leaving"] = abs(merged_section["Ending"].diff())
    colors = ["green", 'orange', 'red']
    fig = px.bar(merged_section, x="index", y=["New", "Charging", "Leaving"],
            barmode='stack')
    fig.update_layout(
        autosize=True,
        width=1000,
        height=400,
        showlegend=True,
        template='plotly_white',
        xaxis_title="Time",
        yaxis_title="Number of cars",
        title="Number of charging cars per day"
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
    section_df = pd.read_pickle("app/static/predictions_timestamp_start.p")

    # graph = px.line(x=section_df.index, y=section_df[section_df.columns[0]])
    graph = px.scatter(x=section_df.index, y=section_df[section_df.columns[0]], trendline="ols")
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

    negative = False
    session_plot = []
    if forecast:
        df, number_of_EVs, number_of_EVs_end = run_model(timestamp_start, timestamp_end)
        session_plot = sessions(timestamp_start, timestamp_end)
        try:
            number_of_EVs, number_of_EVs_end = round(number_of_EVs[0]), round(number_of_EVs_end[0])
        except:
            number_of_EVs, number_of_EVs_end = number_of_EVs, number_of_EVs_end

        if number_of_EVs_end > 0:
            negative = True
    else:
        number_of_EVs, number_of_EVs_end = [], []
    
    print(number_of_EVs, number_of_EVs_end)
    div = main_plot()
    density_plot, scatter_plot, prob = plot_density(hour=timestamp_start.hour, day=timestamp_start.dayofweek)
    observed_plot = observed_data()

    # Temporary!
    number_of_EVs_end = "-"
    return render_template('home.html', 
                            start_date=timestamp_start.strftime("%Y-%m-%d"),
                            end_date=timestamp_end.strftime("%Y-%m-%d"),
                            start_time=str(timestamp_start.hour) + ":00",
                            end_time=str(timestamp_end.hour) + ":00",
                            selected_time=timestamp_start,
                            number_of_EVs=number_of_EVs, 
                            number_of_EVs_end=number_of_EVs_end,
                            negative=negative,
                            plot=div, 
                            prob=prob,
                            bar=session_plot,
                            scatter_plot=scatter_plot,
                            observed_plot=observed_plot,
                            density_plot=density_plot)


@server.route('/forecast/', methods=["POST", "GET"])
def forecast():
    # if request.method == 'GET':

    start_date, start_time = request.form.get("start_date_select"), request.form.get("start_time_select")
    end_date, end_time = request.form.get("end_date_select"), request.form.get("end_time_select")
    
    # error getting data
    if not start_date:
        return predict()

    start_str = start_date + " " + start_time + ':00'
    end_str = end_date + " " + end_time + ':00'

    timestamp_start = pd.to_datetime(start_str)
    timestamp_end = pd.to_datetime(end_str)
    return predict(timestamp_start=timestamp_start, timestamp_end=timestamp_end, forecast=True)

def run_app(host, port):
    server.run()

if __name__ == '__main__':
    server.run(debug=True)



