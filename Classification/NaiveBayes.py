# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 19:29:34 2020

@author: Aditya
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Social_Network_Ads.csv")
print(dataset)

X = dataset.iloc[:, 2:3].values
y = dataset.iloc[:, 4].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
norm = StandardScaler()
X_train = norm.fit_transform(X_train)
X_test = norm.transform(X_test)


from sklearn.naive_bayes import GaussianNB
classification = GaussianNB();
classification.fit(X_train, y_train)


y_pred = classification.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print(acc)


ypred = classification.predict(X_train)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_train,ypred)
print(acc)