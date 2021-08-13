# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:58:58 2020

@author: Aditya
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

dataset=pd.read_csv("./SalaryPrediction.csv")
dataset=dataset.dropna(axis=0,how="any",subset=["Salary","YearsExperience"])


X=dataset.iloc[:,0:1].values        #input
Y=dataset.iloc[:,1].values          #output


#Split Data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=0.2)

Xtrain=Xtrain.reshape(-1,1)
Xtest=Xtest.reshape(-1,1)

#Analyze
reg=LR()
reg.fit(Xtrain,Ytrain)

#Predict
pred=reg.predict(Xtest)
print(r2_score(Ytest, pred))

#plot graph
#plt.scatter(Xtrain,Ytrain, color="red")
#plt.scatter(Xtest,Ytest, color="blue")
#plt.plot(Xtest,pred,color="yellow")
#plt.show()
