# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:52:13 2020

@author: Aditya
"""


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as pp
from sklearn.metrics import r2_score

dataset=pd.read_csv("./Position_Salary.csv")
X=dataset.iloc[:,1].values        #input
Y=dataset.iloc[:,2].values          #output
X=X.reshape(-1,1)

reg1=LR()
reg1.fit(X,Y)
Ypred=reg1.predict(X)

poly=pp(degree=3)
Xpoly=poly.fit_transform(X)

reg=LR()
reg.fit(Xpoly,Y)
Ypredpoly=reg.predict(Xpoly)


plt.scatter(X,Y,color="blue")
plt.plot(X,Ypred,color="green")
plt.plot(X,Ypredpoly,color="red")
plt.show()

pred=reg.predict(poly.transform([[7.8]]))
print(pred)

print(r2_score(Y, Ypredpoly))
