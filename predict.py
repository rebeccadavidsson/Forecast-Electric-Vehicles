import pandas as pd
import numpy as np
import matplotlib.pylab as plt
plt.rcParams["figure.figsize"] = (12,5)
import seaborn as sns
sns.set()

import plotly.express as px
from flask import jsonify
import plotly.figure_factory as ff
import statsmodels
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

from statsmodels.formula.api import ols
from statsmodels.graphics import tsaplots


from random import random

days_in_year = 365 * 24

FUTURE_DAYS = 3000
WINDOW = FUTURE_DAYS + 36


def prepare_dataset():

    weather_df = pd.read_csv("weather.csv", sep=";")
    df = pd.read_csv("charge_sessions_sampled.csv")

    # Convert datetime
    start_date = pd.to_datetime('2019-01-01 00:00')
    weather_df['YYYYMMDD'] = pd.to_datetime(weather_df['YYYYMMDD'], format='%Y%m%d')
    weather_df['Date'] = [start_date + datetime.timedelta(hours=i) for i in range(len(weather_df))]

    # Convert timestamp
    df['session_time'] = df['timestamp_end'] - df['timestamp_start']
    df['timestamp_start'] = pd.to_datetime(df['timestamp_start'], unit='s')
    df['timestamp_end'] = pd.to_datetime(df['timestamp_end'], unit='s')
    df['week_day'] = df['timestamp_start'].dt.dayofweek 
    df["session_time_hours"] = df["session_time"] / 3600
    df = df.assign(Date=df.timestamp_start.dt.floor('H'))
    df = df.assign(Day=df.timestamp_start.dt.floor('D'))
    df["Hour"] = df.Date.dt.hour

    # Exclude outliers
    df = df[df['session_time_hours'] <  50]
    df = df[df['session_time_hours'] > 0.01]
    df = df[df["kwh_charged"] / df["session_time_hours"] < 50]
    df = df[df['kwh_charged'] < 100] 
    df = df[df['kwh_charged'] > 0] 
    # print(len(df))

    counted_df = df.groupby("Date").count()
    mean_df = df.groupby("Date").mean()
    total_rows = len(counted_df)

    # print(df)
    # Merge weather and original df
    df = pd.merge(df, weather_df, on='Date')

    # print(df.head(5))
    return df, total_rows, counted_df, mean_df

def prepare_compact_dataset():

    df = pd.read_csv("charge_sessions_sampled.csv")

    # Convert timestamp
    df['session_time'] = df['timestamp_end'] - df['timestamp_start']
    df['timestamp_start'] = pd.to_datetime(df['timestamp_start'], unit='s')
    df['timestamp_end'] = pd.to_datetime(df['timestamp_end'], unit='s')
    df['week_day'] = df['timestamp_start'].dt.dayofweek 
    df["session_time_hours"] = df["session_time"] / 3600
    df = df.assign(Date=df.timestamp_start.dt.floor('H'))
    df = df.assign(Day=df.timestamp_start.dt.floor('D'))
    df["Hour"] = df.Date.dt.hour

    # Exclude outliers
    df = df[df['session_time_hours'] <  50]
    df = df[df['session_time_hours'] > 0.01]
    df = df[df["kwh_charged"] / df["session_time_hours"] < 50]
    df = df[df['kwh_charged'] < 100] 
    df = df[df['kwh_charged'] > 0] 

    counted_df = df.groupby("Day").count()
    return counted_df


def plot(counted_df):
    plt.plot(counted_df.index, counted_df.session_time, alpha=0.8)
    plt.axvline(x=pd.to_datetime("2020-03-20"), color="black", linestyle=":")
    plt.text(x=pd.to_datetime("2020-03-22"), y=100, s="Start pandemic", color='black')
    plt.ylabel("Frequency of charging cars per HOUR and DAY")

    nr_cars = counted_df[counted_df.index == pd.to_datetime("2021-02-01 14:00")]['session_time'][0]

    plt.axvline(x=pd.to_datetime("2021-02-01 14:00"), color="black", linestyle=":")
    plt.text(x=pd.to_datetime("2021-02-01 14:00"), y=100, s=str(nr_cars) + " cars on 14:00", color='black')
    plt.show()
    print('\033[1m'  + "Number of charging cars on 2021-02-01 14:00:", nr_cars)

    group_by_day = df.groupby("Day").count()

    plt.plot(group_by_day.index, group_by_day.session_time, alpha=0.8)
    plt.axvline(x=pd.to_datetime("2020-03-21"), color="black")
    plt.ylabel("Frequency of charging cars per DAY")
    plt.text(x=pd.to_datetime("2020-03-21"), y=100, s="Start pandemic", color='black')
    plt.show()

    nr_cars = group_by_day[group_by_day.index == pd.to_datetime("2020-03-21")]['session_time'][0]
    print('\033[1m'  + "Total number of charging cars on start of pandemic in the Netherlands:", nr_cars)


def convert_df_corona(counted_df):
    replace_start = pd.to_datetime("2019-03-15")
    replace_end = pd.to_datetime("2019-04-15")

    # replace_with_start = pd.to_datetime("2019-02-15")
    # replace_with_end = pd.to_datetime("2019-04-15")

    new_counted_df = counted_df.copy()
    new_values = counted_df[(counted_df.index >= replace_start) & (counted_df.index < replace_end)]
    new_values.index = new_values.index + datetime.timedelta(days=365)
    new_counted_df = new_counted_df.reset_index()[["Date", "session_time"]]
    new_values = new_values.reset_index()[["Date", "session_time"]]
    converted = pd.merge(new_values, new_counted_df, on="Date", how="outer")
    converted['session_time_x'] = converted['session_time_x'].fillna(converted['session_time_y'])

    converted = converted.set_index("Date")
    converted = converted.drop("session_time_y", axis=1)
    y = converted["session_time_x"]
    # y.plot()
    return y, converted


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)
 

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def fit_model(y, factor):
    series = y * factor
    # seasonal difference
    X = series.values
    differenced = difference(X, days_in_year)
    # fit model
    model = ARIMA(differenced, order=(5, 0, 1))
    model_fit = model.fit()
    return model_fit


def predict_future(y, model_fit, total_rows, timestamp_start, timestamp_end, print_results=False, future_steps=3000, plot=False):
    future_steps = future_steps
    last_date = y.index[-1]

    X = y.values
    # extend index
    new_dates = []
    for t in range(future_steps):
        new_date = last_date + datetime.timedelta(hours=t)
        new_dates.append(new_date)

    forecast = model_fit.forecast(steps=future_steps)

    # invert the differenced forecast to something usable
    history = [x for x in X]
    day = 1
    for yhat in forecast:
        inverted = inverse_difference(history, yhat, days_in_year)
        # print('Day %d: %f' % (day, inverted))
        history.append(inverted)
        day += 1

    factor = len(X) / total_rows

    series = y * factor
    new_index = list(series.index) + new_dates
    predictions = np.asarray(history) / np.asarray(factor)

    results_df = pd.DataFrame(predictions, index=new_index)
    print(timestamp_start, timestamp_end, "TIMESTAMP")
    number_of_EVs = results_df[results_df.index == timestamp_start][0]
    number_of_EVs_end = results_df[results_df.index == timestamp_end][0]

    if print_results:
        print("Number of EVs predicted for this sample ", round(number_of_EVs.values[0]))

        difference = round(number_of_EVs.values[0] - number_of_EVs_end.values[0])
        if difference < 0:
            print("Number of EVs discharged", abs(round(number_of_EVs)))
        else:
            print("Number of NEW EVs charging", round(difference))

    if plot:
        section_index, section = new_index[-3000:], predictions[-3000:]
        plt.plot(section_index, section, color='orange', label="Predictions")
        plt.axvline(x=pd.to_datetime('2021-04-04'), linestyle="-", color="black")
        plt.text(x=pd.to_datetime('2021-04-04'), y=np.max(section), s=str(int(number_of_EVs.values[0])) + " cars")
        plt.ylabel("Frequency of charging cars per HOUR and DAY")
        plt.legend()
        plt.show()

    return number_of_EVs.values[0], number_of_EVs_end.values[0]
    

def run(timestamp_start, timestamp_end):
    
    print("Predicting...")
    df, total_rows, counted_df, mean_df = prepare_dataset()
    y = counted_df["timestamp_start"]
    y_mean = mean_df["session_time"]


    y, converted = convert_df_corona(counted_df)
    y_mean, converted_mean = convert_df_corona(mean_df)

    factor = len(y) / total_rows
    model_fit = fit_model(y, factor)
    model_fit_mean = fit_model(y_mean, factor)

    number_of_EVs, number_of_EVs_end = predict_future(y, model_fit, total_rows, timestamp_start, timestamp_end, plot=False)
    # number_of_EVs_, number_of_EVs_end_ = predict_future(y_mean, model_fit_mean, total_rows, timestamp_start, timestamp_end, plot=False)

    return number_of_EVs, number_of_EVs_end


def weather():
    df, total_rows, counted_df, _ = prepare_dataset()
    y, converted = convert_df_corona(counted_df)
    
    model1 = ols('kwh_charged ~ rain + covid_shock', data=df).fit() 
    model2 = ols('session_time ~ rain + covid_shock + wind + tunder + ice ', data=df).fit() 

    fig = tsaplots.plot_acf(y, lags=7)
    plt.show()
    print(fig.show())

    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='seismic', alpha=0.8)
    
    X = df['Date']
    Y= df['session_time']
    Z= df['kwh_charged']
    fig = plt.figure(figsize = (10, 10))
    ax= fig.gca(projection="3D")
    ax.plot_surface()
    

if __name__ == '__main__':
    # weather()
    number_of_EVs, number_of_EVs_end = run()
    #plot_density()
    #fit_modelL()

