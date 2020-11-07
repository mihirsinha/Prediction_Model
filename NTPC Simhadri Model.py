# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 22:45:41 2020

@author: h
"""
"""
Performance - 
Epoch 285/285
20/20 [==============================] - 9s 431ms/step - loss: 0.1098
"""
import pandas as pd
import numpy as np
import datetime as dt
import glob,os
from datetime import timedelta, date
from math import sqrt
from numpy import array
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

dates = ["{0:0=2d}".format(i) for i in range(1,32)]

# Date Range
def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)
datelist = list()
start_dt = date(2020, 7, 1)
end_dt = date(2020, 7, 31)
for dt in daterange(start_dt, end_dt):
    datelist.append(dt.strftime("%d-%m-%Y"))

# Number of Revisions for each day
number_of_files = []
for d in dates:
    path = 'C:/Users/h/Desktop/NTPC/Seller Side-Simhadri/{}-07-2020/'.format(d) 
    all_files = sorted(glob.glob(os.path.join(path, "*.csv")))
    number_of_files.append(len(all_files))
    
# Combining each combined csv files into one and reshaping it to form 3D dataframe
filepath = 'C:/Users/h/Desktop/NTPC/Combined CSV by date/' 
all_files = sorted(glob.glob(os.path.join(filepath, "*.csv")))     # advisable to use os.path.join as this makes concatenation OS independent
#files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[0]))
df_from_each_file = (pd.read_csv(f, encoding = 'utf-8') for f in all_files)
concatenated_df   = pd.concat(df_from_each_file, axis=0)
data_array = concatenated_df.drop([concatenated_df.columns[0], concatenated_df.columns[1]], axis = 1)
data_array = np.array(data_array)
data_array = np.reshape(data_array, (len(number_of_files), 96, data_array.shape[1]))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scalers = {}
for i in range(data_array.shape[0]):
    scalers[i] = StandardScaler()
    data_array[i, :, :] = scalers[i].fit_transform(data_array[i, :, :]) 

# Split into train and test
train = data_array[:21][:][:]
test = data_array[21:][:][:]

# Evaluate model and get scores
n_input = 96
n_out=96
## to_supervised
data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
X, y = list(), list()
in_start = 0

# step over the entire history one time step at a time
for _ in range(len(data)):
    # define the end of the input sequence
    in_end = in_start + n_input
    out_end = in_end + n_out
    # ensure we have enough data for this instance
    if out_end <= len(data):
        x_input = data[in_start:in_end, :]
        #x_input = x_input.reshape((len(x_input), 1))
        X.append(x_input)
        y.append(data[in_end:out_end, :])
    # move along one time step
    in_start += 96
train_x, train_y = array(X), array(y)

epochs, batch_size = 285, 1
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
# define model
model = Sequential()
model.add(LSTM(200, return_sequences=True, input_shape=(n_timesteps, n_features)))
model.add(Dropout(0.3))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.3))
model.add(Dense(units = max(number_of_files)))
model.compile(loss='mse', optimizer='adam')
# fit network
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size)

"""
Epoch 285/285
20/20 [==============================] - 8s 402ms/step - loss: 0.1963
"""

history = [x for x in train]
predictions = list()	

def forecast(model, history, n_input):
    data = array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    input_x = data[-n_input:, :]
    #print(n_input)
    input_x = input_x.reshape((1, len(input_x), max(number_of_files)))
    yhat = model.predict(input_x)
    yhat = yhat[0]
    return yhat

for i in range(len(test)):
	# predict the week
	yhat_sequence = forecast(model, history, n_input)
	# store the predictions
	predictions.append(yhat_sequence)
	# get real observation and add to history for predicting the next week
	history.append(test[i, :])
# evaluate predictions days for each week
predictions = array(predictions)

n_lambda, n_alpha, n_beta = test.shape[0], 96, test.shape[2]
error_matrix = np.zeros([n_lambda, n_alpha, n_beta], dtype = float)
for i in range(n_lambda):
    for j in range(n_alpha):
        for k in range(n_beta):
            error_matrix[i,j,k] = sqrt(mean_squared_error([test[i,j,k]], [predictions[i,j,k]]))

# Inverse Transforming the predicted and actual values for better understanding.
predicted = np.zeros([n_lambda, n_alpha, n_beta], dtype = float)
actual = np.zeros([n_lambda, n_alpha, n_beta], dtype = float)
for i in range(predictions.shape[0]):
    #scalers[i] = StandardScaler()
    predicted[i, :, :] = scalers[i].inverse_transform(predictions[i, :, :]) 

for i in range(test.shape[0]):
    #scalers[i] = StandardScaler()
    actual[i, :, :] = scalers[i].inverse_transform(test[i, :, :]) 
    
# Saving the Trained LSTM Model
import pickle as p
file_name = "NTPC_SimhadriPlant_Model[30 Days].pkl"
p.dump(model, open(file_name, "wb"))  #Save



