# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:21:25 2020

@author: Aviral
"""

##############################################################################
"""
After Training the Model and saving it, directly importing it and using it...
"""
##############################################################################

# Importing Pickle and Loading saved model
import pickle as p
file_name = "NTPC_SimhadriPlant_Model[30 Days].pkl"
loaded_model = p.load(open(file_name, "rb"))

# Importing the libraries
import pandas as pd
import numpy as np
import datetime as dt
import glob,os
from datetime import timedelta, date
from math import sqrt
from numpy import array
from sklearn.metrics import mean_squared_error

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

# Making predictions for the next month based on 1 month on input    
pred = loaded_model.predict(data_array)
# Inverse transforming the values for better understanding.
transformed_pred = pred
for i in range(pred.shape[0]):
    #scalers[i] = StandardScaler()
    transformed_pred[i, :, :] = scalers[i].inverse_transform(pred[i, :, :]) 
