
from typing import final
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from keras.models import load_model
import streamlit as st
plt.style.use('fivethirtyeight')
import yfinance as yf
yf.pdr_override()
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.svm import SVR

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker','AAPL,GOOG,AMZN,NFLX')
start= st.text_input('Enter Date From','2018-01-01')
end= st.text_input('Enter Date To','2020-12-30')
stockSymbols = user_input.split(",")

#Describing Data
#Analyst
stock_list = list()
for stock in stockSymbols:
    globals()[stock] = yf.download(stock, start, end)

stock_name = stockSymbols

for stock in stockSymbols:
    stock_list.append(globals()[stock])
    st.subheader('Data ' + stock)
    globals()[stock]
    # st.subheader('Describe data ' + stock)
    # st.write(globals()[stock].describe())
    fig_stock=plt.figure(figsize=(12.2,4.5))
    plt.plot(globals()[stock]["Adj Close"],label=stock)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price USD($)',fontsize=18)
    st.pyplot(fig_stock)


 
  



df = pdr.get_data_yahoo(stockSymbols, start=start, end=end)
st.subheader('Data from '+start+' - '+end)
st.write(df)

df = df["Adj Close"]
st.subheader('Close Price from '+start+' - '+ end)
st.write(df)

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
fig_heatmap_corr=plt.figure(figsize=(12.2,4.5))
sns.heatmap(corr, annot=True, cmap='coolwarm')
st.pyplot(fig_heatmap_corr)

#Show The covariance matrix for simple returns
st.subheader('The covariance matrix for simple returns')
cov=daily_simple_returns.cov()
cov
fig_heatmap_cov=plt.figure(figsize=(12.2,4.5))
sns.heatmap(cov, annot=True, cmap='coolwarm')
st.pyplot(fig_heatmap_cov)
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
# dailyMeanSimpleReturns=daily_simple_returns.mean()
# #Print
# st.subheader("The daily mean simple return:")
# dailyMeanSimpleReturns

#5. How much value do we put at risk by investing in a particular stock?
#There are many ways we can quantify risk, one of the most basic ways using the information we've gathered on daily percentage returns is by comparing the expected return with the standard deviation of the daily returns.
# Let's start by defining a new DataFrame as a cleaned version of the original tech_rets DataFrame
st.subheader("Risk and daily mean simple return")
rets = daily_simple_returns.dropna()

area = np.pi * 20

expected_risk_return_fig = plt.figure(figsize=(10, 7))
plt.scatter(rets.mean(), rets.std(), s=area)
plt.xlabel('Expected return')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', 
                 arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))
st.pyplot(expected_risk_return_fig)
#End Analyst
##############################################

#Predict ################################################################

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


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

 #Predicting the closing price stock price of APPLE inc:
 # Create a new dataframe with only the 'Close column 
data = stock_selected2.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values

# Get the number of rows to train the model on
st.subheader("Get the number of rows to train the model on 80%")
training_data_len = int(np.ceil(len(dataset) * .8 ))
training_data_len

# Scale the data
st.subheader("Scale the data")
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
st.subheader("  Get the root mean squared error")
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse

# Plot the data LSTM Model
st.subheader("  LSTM Model")
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

st.subheader("  Predict Value using LSTM Model")
st.write(valid)

#Get only the dates and the adjusted close prices
days = list()
close_prices = list()
user_input_2 = st.text_input('Enter Stock','AAPL')
df2 = pdr.get_data_yahoo(user_input_2, start=start, end=end)
#Show data from 01-30 month 12 2020
st.subheader('Data from 2020-12-01 - 2020-12-30')
df2


#Get all of the data except the last row
#Show the data set
df2.head()
df2 = df2.tail(100)
df2 = df2.reset_index()
data_train = df2.head(90)
#Show and store the last row of data
actual_price = df2.tail(10)
#Get only the dates and the  close prices
df_days = data_train.loc[:, 'Date']
df_close = data_train.loc[:, 'Close']
df2


#Create the independent data set (dates)
# for day in df_days:
#   days.append([int(day.strftime ("%d"))])
for i in range(len(df_days)):
  days.append([i+1])
#Create the dependent data set ( close prices)
for close_price in df_close:
  close_prices.append(float(close_price))
  
#Create 3 models
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
number_days_input = st.text_input('Enter the number of next predicted days','10')

st.subheader('Predicted price for the next ' +number_days_input+ ' days')

arr_days_predict = list()

for i in range(int(number_days_input)):
  arr_days_predict.append([i+1+len(days)])

st.subheader('The actual price:')
for vl in actual_price['Close']:
  st.write(vl)

st.subheader('The RBF SVR predicted price:')
for vl in rbf_svr.predict(arr_days_predict):
  st.write(vl)

st.subheader('The Linear SVR predicted price:')
for vl in lin_svr.predict(arr_days_predict):
  st.write(vl)

st.subheader('The Polynomial SVR predicted price:')
for vl in poly_svr.predict(arr_days_predict):
  st.write(vl)

# html_str = f"""
# <h4>The actual price:{actual_price['Close']}</h4>
# <h4>The RBF SVR predicted price:{rbf_svr.predict(arr_days_predict)}</h4>
# <h4>The Linear SVR predicted price:{lin_svr.predict(arr_days_predict)}</h4>
# <h4>The Polynomial SVR predicted price:{poly_svr.predict(arr_days_predict)}</h4>
# """

# st.markdown(html_str, unsafe_allow_html=True)
