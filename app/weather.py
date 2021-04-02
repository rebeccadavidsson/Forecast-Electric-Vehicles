import pandas as pd
from random import random

days_in_year = 365 * 24


def prepare_dataset():
    df = pd.read_csv("charge_sessions_sampled.csv")
    weather_df = pd.read_csv("weather.csv")
    print(weather_df.head(5))
    # Convert timestamp
    df['session_time'] = df['timestamp_end'] - df['timestamp_start']
    df['timestamp_start'] = pd.to_datetime(df['timestamp_start'], unit='s')
    df['timestamp_end'] = pd.to_datetime(df['timestamp_end'], unit='s')
    df['week_day'] = df['timestamp_start'].dt.dayofweek 
    df["session_time_hours"] = df["session_time"] / 3600
    df = df.assign(Date=df.timestamp_start.dt.floor('H'))
    df = df.assign(Day=df.timestamp_start.dt.floor('D'))

    # Exclude outliers
    df = df[df['session_time_hours'] <  50]
    df = df[df['session_time_hours'] > 0.01]
    df = df[df["kwh_charged"] / df["session_time_hours"] < 50]
    df = df[df['kwh_charged'] < 100] 
    df = df[df['kwh_charged'] > 0] 
    # print(len(df))

    counted_df = df.groupby("Date").count()
    total_rows = len(counted_df)
    # print("total rows", total_rows)

    return df, total_rows, counted_df


prepare_dataset()
