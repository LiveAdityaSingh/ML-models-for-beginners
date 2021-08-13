# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 22:08:48 2020

@author: Aditya
"""


import pandas as pd
import numpy as np

dataset=pd.read_csv("./Social_Network_Ads.csv")
X=dataset.iloc[:,1:4].values        #input
Y=dataset.iloc[:,4].values          #output

from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
labelEncoder.fit(X[:,0])
X[:, 0] = labelEncoder.transform(X[:,0])

from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X, Y,test_size=0.2, random_state=1)

from sklearn.svm import SVC
classi=SVC()
classi.fit(Xtrain,Ytrain)

y_pred=classi.predict(Xtest)

from sklearn.metrics import accuracy_score
acc=accuracy_score(Ytest, y_pred)
print(acc)