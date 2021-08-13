# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:13:30 2020

@author: Aditya
"""
import pandas as pd
import numpy as np

dataset=pd.read_csv("./Social_Network_Ads.csv")
X=dataset.iloc[:,2:3].values        #input
Y=dataset.iloc[:,4].values          #output

from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X, Y,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
cassi=KNeighborsClassifier(n_neighbors=41 , metric='euclidean')
cassi.fit(Xtrain,Ytrain)

ypred=cassi.predict(Xtest)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Ytest, ypred)
print(cm)

from sklearn.metrics import accuracy_score
acc=accuracy_score(Ytest, ypred)
print(acc)

