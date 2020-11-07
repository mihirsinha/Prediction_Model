# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 21:41:16 2020

@author: h
"""

"""
Performance while training - 
Epoch 200/200
5/5 [==============================] - 0s 27ms/step - loss: 0.0224
"""
# Importing Libraries
import pandas as pd
import numpy as np
from math import sqrt
from numpy import array
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Importing the Dataset
dataset = pd.read_excel('India Demand.xlsx')
dataset = dataset.iloc[::-1].reset_index(drop=True)
dataset = dataset.drop(['gendate'],axis = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
ScaledData = sc.fit_transform(dataset)

# Reshaping Scaled Data into 3-D dataframe
ScaledData = np.array(ScaledData)
DataDate = ScaledData.reshape(9,3,96) # RESHAPE ACCORDINGLY

# Splitting into train and test data based on Dates
train = DataDate[:6][:][:]
test = DataDate[6:][:][:]

# Splitting Train data further into train_x [19/08 - 02/09] and train_y [22/08 - 05/09] 
n_input = 3 # Slab of input consisting of 3 days
n_out=3     # Slab of output consisting of 3 days
in_start = 0
X, y = list(), list()
data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
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
    in_start += 3
train_x, train_y = array(X), array(y)

# Define LSTM Sequential model
epochs, batch_size = 200, 1
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

model = Sequential()
# Adding first layer of lstm
model.add(LSTM(200, return_sequences=True, input_shape=(n_timesteps, n_features)))
model.add(Dropout(0.3))
# Adding second layer of lstm
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.3))
# Adding third layer of lstm
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.3))
# Adding fourth layer of lstm
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.3))
# Adding output layer of lstm
model.add(Dense(units = 96))
# Compiling all the layers
model.compile(loss='mse', optimizer='adam')
# Fitting the model to the training data
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size)

# History - list of values according to the dates
history = [x for x in train]
predictions = list()	

# Function to predict the values based on input values from history
def forecast(model, history, n_input):
    data1 = array(history)
    data1 = data1.reshape((data1.shape[0]*data1.shape[1], data1.shape[2]))
    input_x = data1[-n_input:, :]
    print(n_input)
    input_x = input_x.reshape((1, len(input_x), 96))
    yhat = model.predict(input_x)
    yhat = yhat[0]
    return yhat

# Applying the model to Test data
for i in range(len(test)):
	# predict the next 3 days
	yhat_sequence = forecast(model, history, n_input)
	# store the predictions
	predictions.append(yhat_sequence)
	# get real observation and add to history for predicting the next week
	history.append(test[i, :])

# Converting predictions into an array and reshaping it
predictions = array(predictions)
predicted = predictions.reshape(predictions.shape[0]*predictions.shape[1],predictions.shape[2])
actual = test.reshape(test.shape[0]*test.shape[1],test.shape[2])

# Calculating mse by cmparing actual and predicted values
n_lambda, n_alpha = test.shape[0]*test.shape[1], 96
error_matrix = np.zeros([n_lambda, n_alpha], dtype = float)
for i in range(n_lambda):
    for j in range(n_alpha):
        error_matrix[i, j] = sqrt(mean_squared_error([actual[i,j]], [predicted[i, j]]))
        

""" Only for Experimentation Purpose [lines 131-150] """
# evaluate all forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[0]):
		# calculate mse
		mse = mean_squared_error(array(actual[i,:]), array(predicted[i,:]))
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores
score, scores = evaluate_forecasts(actual, predicted)

# Inverse Transforming the predicted and actual values for better understanding.
predicted = sc.inverse_transform(predicted)
actual = sc.inverse_transform(actual)

# Saving the Trained LSTM Model
import pickle as p
file_name = "NTPC_Overall_Model[27 Days].pkl"
p.dump(model, open(file_name, "wb"))  #Save
