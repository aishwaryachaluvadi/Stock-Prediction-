import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from kats.consts import TimeSeriesData
import plotly
from datetime import datetime
import io
import plotly.graph_objects as go
from datetime import datetime, date
import plotly.graph_objects as go

col1,col2,col3 = st.columns([2,1,2])
with col1:
    st.image("woxsen_logo.PNG")

with col3:
    st.image("appstek_logo.jpg")

st.title("Stock Prediction")

def file_uploader():

    uploaded_file = st.file_uploader("Upload the csv file")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write(data)
        columns = list(data.columns)
        columns.append("-")

        time_column = st.selectbox("Select the time column", columns[::-1], key = "a")
        value_column = st.selectbox("Select the value column to predict", columns[::-1], key = "b")

        df = pd.DataFrame(columns = ["time", "value"])

        if time_column != "-" and value_column != "-":
            df["time"] = data[time_column]
            df["value"] = data[value_column]
            

            df_ts = TimeSeriesData(df, use_unix_time=True)
            # import the param and model classes for Prophet model
            from kats.models.prophet import ProphetModel, ProphetParams

            # create a model param instance
            params = ProphetParams(seasonality_mode='multiplicative') # additive mode gives worse results

            # create a prophet model instance
            m = ProphetModel(df_ts, params)

            # fit model simply by calling m.fit()
            m.fit()

            new_date = st.date_input("Enter a Date to Predict", datetime.now())
            df['time'] = pd.to_datetime(df['time'])
            val = df["time"][len(df)-1].strftime('%Y-%m-%d')
            old_date = datetime.strptime(val,"%Y-%m-%d")

            days = (new_date - old_date.date()).days

            if st.button("Predict"):
                fcst = m.predict(steps=days, freq="D")
                m.plot()
                st.write(fcst.tail(1))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["time"], y=df["value"],
                                    mode='lines',
                                    name='lines'))
                fig.add_trace(go.Scatter(x=fcst["time"], y=fcst["fcst"],
                                    mode='lines+markers',
                                    name='lines+markers'))
                st.write("Plotting the Data")
                st.write(fig)

                
        


if __name__ == "__main__":
    file_uploader()
