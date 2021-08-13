"""
Created on Thu Jun 11 19:13:45 2020

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

from sklearn.tree import DecisionTreeClassifier as dt
classi=dt(criterion="entropy")
classi.fit(Xtrain,Ytrain)

ypred=classi.predict(Xtest)
print(ypred)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Ytest,ypred)
print(cm)

from sklearn.metrics import accuracy_score
acc=accuracy_score(Ytest,ypred)
print(acc)

from sklearn.ensemble import RandomForestClassifier
rc= RandomForestClassifier(n_estimators=200,criterion="entropy")
rc.fit(Xtrain,Ytrain)

ypredr=rc.predict(Xtest)
print(ypredr)

cmr=confusion_matrix(Ytest,ypredr)
print(cmr)
accr=accuracy_score(Ytest,ypredr)
print(accr)