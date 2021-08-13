# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 19:31:19 2020

@author: Aditya
"""


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as lr
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler as ss

dataset=pd.read_csv("./Social_Network_Ads.csv")
gender=dataset.iloc[:,1:2].values
X=dataset.iloc[:,2:4].values        #input
Y=dataset.iloc[:,4].values          #output

labelEncoder = LabelEncoder()
labelEncoder.fit(gender[:,0])
gender[:, 0] = labelEncoder.transform(gender[:,0])


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=0.2)
X=X.reshape(-1,1)

nrm=ss()
Xtrain=nrm.fit_transform(Xtrain)
Xtest=nrm.transform(Xtest)

reg=lr()
reg.fit(Xtrain,Ytrain)
y_pred=reg.predict(Xtest)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Ytest, y_pred)
print(cm)






