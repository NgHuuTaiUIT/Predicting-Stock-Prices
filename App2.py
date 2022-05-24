from typing import final
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from keras.models import load_model
import streamlit as st
plt.style.use('fivethirtyeight')


st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker','AAPL,GOOG,AMZN,NFLX,FB')
start= st.text_input('Enter Date From','2013-01-01')
end= st.text_input('Enter Date To','2020-01-28')
stockSymbols = user_input.split(",")

df = pdr.get_data_yahoo(stockSymbols, start=start, end=end)
#Describing Data

st.subheader('Data from '+start+' - '+end)
st.write(df.describe())

df = df["Close"]
st.subheader('Close Price from '+start+' - '+ end)
st.write(df.describe())

#Createafunction to visualize the portfolio
st.subheader('Portfolio Close Price History')
#Get the stocks
my_stocks=df
#Give the figure size
fig_my_stocks=plt.figure(figsize=(12.2,4.5))
#Loop through each stock and plot the price
for c in my_stocks.columns.values:
    plt.plot(my_stocks[c],label=c)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD($)',fontsize=18)
plt.legend(my_stocks.columns.values,loc='upper left')
st.pyplot(fig_my_stocks)

#Calculate the simple returns
st.subheader('Calculate the simple returns')
my_stocks = df
daily_simple_returns=my_stocks.pct_change(1)
#Show the daily simple returns
daily_simple_returns
#Show the stock correlation
st.subheader('The stock correlation')
corr=daily_simple_returns.corr()
corr
#Show The covariance matrix for simple returns
st.subheader('The covariance matrix for simple returns')
cov=daily_simple_returns.cov()
cov
#Show the variance
st.subheader('The variance')
var=daily_simple_returns.var()
var
#Print the standard deviation for daily simple returns
st.subheader('The standard deviation for daily simple returns')
std = daily_simple_returns.std()
std
#Visualize the stocks daily simple returns/Volatility
st.subheader('The stocks daily simple returns/Volatility')
fig_volatility = plt.figure(figsize=(12,4.5))
#Loop through each stock and plot the simple returns
for i in daily_simple_returns.columns.values:
  plt.plot(daily_simple_returns[i],lw=2,label=i)
plt.legend(loc='upper right',fontsize=10)
plt.title('Volatility')
plt.ylabel('Daily Simple Returns')
plt.xlabel('Date')
st.pyplot(fig_volatility)

#Show the mean of the daily simple return
dailyMeanSimpleReturns=daily_simple_returns.mean()
#Print
st.subheader("The daily mean simple return:")
dailyMeanSimpleReturns

#Decision Tree
stock_sl = st.text_input('Enter Stock Ticker','AAPL')
st.subheader('Data ' +stock_sl +' from '+start+' - '+end)
stock_selected = pdr.get_data_yahoo(stock_sl, start=start, end=end)

stock_selected
stock_selected.shape
# Visualizing the close prices of the data.
st.subheader("Close "+stock_sl+" Price History")
stock_selected_fig = plt.figure(figsize=(16,8))
plt.title('Apple')
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(stock_selected['Close'])
st.pyplot(stock_selected_fig)

stock_selected = stock_selected["Close"]
stock_selected = pd.DataFrame(stock_selected)

# Prediction 100 days into the future.
future_days = 100
stock_selected['Prediction'] = stock_selected.shift(-future_days)
x = np.array(stock_selected.drop(['Prediction'], 1))[:-future_days]
y = np.array(stock_selected['Prediction'])[:-future_days]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# Implementing Linear and Decision Tree Regression Algorithms.
tree = DecisionTreeRegressor().fit(x_train, y_train)
lr = LinearRegression().fit(x_train, y_train)

x_future = stock_selected.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)

#Predict Tree
tree_prediction = tree.predict(x_future)
tree_prediction

predictions = tree_prediction 
valid = stock_selected[x.shape[0]:]
valid['Predictions'] = predictions
st.subheader("Predict Tree")
tree_fig = plt.figure(figsize=(16,8))
plt.title("Model")
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(stock_selected['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(["Original", "Valid", 'Predicted'])
st.pyplot(tree_fig)

#Predict Linear
lr_prediction = lr.predict(x_future)
lr_prediction

predictions = lr_prediction 
valid = stock_selected[x.shape[0]:]
valid['Predictions'] = predictions
st.subheader("Predict Linear")
linear_fig = plt.figure(figsize=(16,8))
plt.title("Model")
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(stock_selected['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(["Original", "Valid", 'Predicted'])
st.pyplot(linear_fig)


#Prediction using LSTM
st.subheader("Prediction using LSTM")
stock_selected2 =  pdr.get_data_yahoo('AAPL', start=start, end=end)

#moving average of the various stocks
ma_day = [10, 30, 100]
st.subheader("Moving average of the various stocks")
ma_fig = plt.figure(figsize = (12,6))
for ma in ma_day:
  column_name = f"MA for {ma} days"
  ma_predict = stock_selected2.Close.rolling(ma).mean()
  plt.plot(ma_predict)
plt.plot(stock_selected2.Close)
plt.legend(['MA for 10 days', 'MA for 30 days', 'MA for 100 days','Close'])
st.pyplot(ma_fig)

 #Predicting the closing price stock price of APPLE inc:
 # Create a new dataframe with only the 'Close column 
data = stock_selected2.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .8 ))

training_data_len

# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data

# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse

# Plot the data LSTM Model
st.subheader("LSTM Model")
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
lstm_fig = plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
st.pyplot(lstm_fig)

#Visualization
# st.subheader("Closing Price vs Time chart")
# fig = plt.figure(figsize = (12,6))
# plt. plot (df.Close)
# st.pyplot(fig)

# st.subheader('Closing Price vs Time Chart with 100MA')
# ma100 = df.Close.rolling(100).mean()
# fig = plt.figure(figsize=(12,6))
# plt.plot(df.Close)
# plt.plot(ma100, 'r')
# st.pyplot(fig)

# st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
# ma100 = df.Close.rolling(100).mean()
# ma200 = df.Close.rolling(200).mean()
# fig = plt.figure(figsize = (12,6))
# plt.plot(ma100,'r')
# plt.plot(ma200,'g')
# plt.plot(df.Close)
# st.pyplot(fig)

# # Splitting Data into Training and Testing

# data_traning = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
# data_testing = pd.DataFrame(df[ 'Close' ][int(len(df)*0.70): int(len(df))]) 
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0,1))

# data_training_arr = scaler.fit_transform(data_traning)

# x_train = []
# y_train = []

# for i in range(100, data_training_arr.shape[0]):
#     x_train.append(data_training_arr[i-100: i])
#     y_train.append(data_training_arr[i,0])

# x_train, y_train = np.array(x_train), np.array(y_train)

# #Load my model
# model = load_model('./keras_model.h5')

# #Testing Part

# past_100_days = data_traning.tail(100)
# final_df = past_100_days.append(data_testing, ignore_index=True)
# input_data = scaler.fit_transform(final_df)

# x_test = []
# y_test = []

# for i in range(100, input_data.shape[0]):
#   x_test.append(input_data[i-100: i])
#   y_test.append(input_data[i, 0])

# x_test, y_test = np.array(x_test), np.array(y_test)
# y_predicted = model.predict(x_test)
# scaler = scaler.scale_

# scale_factor = 1/scaler[0]
# y_predicted = y_predicted * scale_factor
# y_test = y_test * scale_factor

# #Final Graph
# st.subheader("Predictions vs Original")
# fig2 = plt.figure(figsize=(12,6))
# plt.plot(y_test, label = 'Original Price')
# plt.plot(y_predicted, 'r', label = 'Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# st.pyplot(fig2)

#Get only the dates and the adjusted close prices
days = list()
close_prices = list()
user_input_2 = st.text_input('Enter Stock','AAPL')
df2 = pdr.get_data_yahoo(user_input_2, start='2021-12-02', end='2021-12-30')
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
#Get only the dates and the  close prices
df_days = df2.loc[:, 'Date']
df_close = df2.loc[:, 'Close']

#Create the independent data set (dates)
for day in df_days:
  days.append([int(day.strftime ("%d"))])
#Create the dependent data set ( close prices)
for close_price in df_close:
  close_prices.append(float(close_price))
  
#Create 3 models
from sklearn.svm import SVR
lin_svr = SVR(kernel='linear', C= 1000.0)
lin_svr.fit(days, close_prices)

poly_svr = SVR(kernel='poly', C= 1000.0, degree=2)
poly_svr.fit(days, close_prices)

rbf_svr = SVR(kernel='rbf', C= 1000.0, gamma=0.85)
rbf_svr.fit(days, close_prices)

#Plot the models
st.subheader('RBF, Polynomial & Linear Models')
fig3=plt.figure(figsize=(16,8))
plt.scatter(days, close_prices, color = 'black', label = 'Data')
plt.plot(days, rbf_svr.predict(days), color = 'green', label = 'RBF Model')
plt.plot(days, poly_svr.predict(days), color = 'orange', label = 'Polynomial Model')
plt.plot(days, lin_svr.predict(days), color = 'blue', label = 'Linear Model')
plt.xlabel('Days')
plt.ylabel('Close Price ($)')
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
