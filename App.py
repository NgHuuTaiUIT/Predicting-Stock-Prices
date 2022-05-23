
from typing import final
from typing_extensions import dataclass_transform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from keras.models import load_model
import streamlit as st
plt.style.use('fivethirtyeight')

start='2012-01-01'
end='2020-01-01'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker','AAPL')
# df = pdr.get_data_yahoo('AAPL', 'yahoo', start, end)
df = pdr.get_data_yahoo(user_input, start, end)
#Describing Data

st.subheader('Data from 2012 - 2019')
st.write(df.describe())


#Visualization
st.subheader("Closing Price vs Time chart")
fig = plt.figure(figsize = (12,6))
plt. plot (df.Close)
st.pyplot(fig)
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close)
st.pyplot(fig)

# Splitting Data into Training and Testing

data_traning = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df[ 'Close' ][int(len(df)*0.70): int(len(df))]) 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_arr = scaler.fit_transform(data_traning)

x_train = []
y_train = []

for i in range(100, data_training_arr.shape[0]):
    x_train.append(data_training_arr[i-100: i])
    y_train.append(data_training_arr[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

#Load my model
model = load_model('./keras_model.h5')

#Testing Part

past_100_days = data_traning.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100: i])
  y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader("Predictions vs Original")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

#Get only the dates and the adjusted close prices
days = list()
adj_close_prices = list()
df2 = pdr.get_data_yahoo(user_input, start='2020-12-02', end='2020-12-30')
#Show data from 01-30 month 12 2020
st.subheader('Data from 2020-12-01 - 2020-12-30')
df2

#Show and store the last row of data
actual_price = df2.tail(1)

#Get all of the data except the last row
df2= df2.head(len(df2)-1)
#Show the data set
df2.head()
df2.tail()
df2 = df2.reset_index()
#Get only the dates and the adjusted close prices
df_days = df2.loc[:, 'Date']
df_adj_close = df2.loc[:, 'Adj Close']

#Create the independent data set (dates)
for day in df_days:
  days.append([int(day.strftime ("%d"))])
#Create the dependent data set (adj close prices)
for adj_close_price in df_adj_close:
  adj_close_prices.append(float(adj_close_price))
  
#Create 3 models
from sklearn.svm import SVR
lin_svr = SVR(kernel='linear', C= 1000.0)
lin_svr.fit(days, adj_close_prices)

poly_svr = SVR(kernel='poly', C= 1000.0, degree=2)
poly_svr.fit(days, adj_close_prices)

rbf_svr = SVR(kernel='rbf', C= 1000.0, gamma=0.85)
rbf_svr.fit(days, adj_close_prices)

#Plot the models
st.subheader('RBF, Polynomial & Linear Models')
fig3=plt.figure(figsize=(16,8))
plt.scatter(days, adj_close_prices, color = 'black', label = 'Data')
plt.plot(days, rbf_svr.predict(days), color = 'green', label = 'RBF Model')
plt.plot(days, poly_svr.predict(days), color = 'orange', label = 'Polynomial Model')
plt.plot(days, lin_svr.predict(days), color = 'blue', label = 'Linear Model')
plt.xlabel('Days')
plt.ylabel('Adj Close Price ($)')
plt.legend()
st.pyplot(fig3)

#Show the predicted price for the given day
st.subheader('Predicted price for 30')

day = [[30]]
html_str = f"""
<h4>The RBF SVR predicted price:{rbf_svr.predict(day)}</h4>
<h4>The Linear SVR predicted price:{lin_svr.predict(day)}</h4>
<h4>The Polynomial SVR predicted price:{poly_svr.predict(day)}</h4>
"""

st.markdown(html_str, unsafe_allow_html=True)
