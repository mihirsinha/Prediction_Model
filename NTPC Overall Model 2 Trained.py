# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:38:14 2020

@author: h
"""

##############################################################################
"""
After Training the Model and saving it, directly importing it and using it...
"""
##############################################################################

# Importing Pickle and Loading saved model
import pickle as p
file_name = "NTPC_Overall_Model[27 Days].pkl"
loaded_model = p.load(open(file_name, "rb"))

# Importing Libraries
import pandas as pd
import numpy as np

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

# Making predictions for the next month based on 1 month on input    
pred = loaded_model.predict(DataDate)

# Reshaping and Inverse transforming the values for better understanding.
transformed_pred = pred.reshape(pred.shape[0]*pred.shape[1],pred.shape[2])
transformed_pred = sc.inverse_transform(transformed_pred)
