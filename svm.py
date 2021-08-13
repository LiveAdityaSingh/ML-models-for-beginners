# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 19:24:20 2020

@author: Aditya
"""



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import r2_score

dataset=pd.read_csv("./Position_Salary.csv")
X=dataset.iloc[:,1].values        #input
Y=dataset.iloc[:,2].values          #output

X=X.reshape(-1,1)
Y=Y.reshape(-1,1)
plt.scatter(X,Y,color="blue")

#nrmX=ss()
#X=nrmX.fit_transform(X)
nrmY=ss()
Y=nrmY.fit_transform(Y)
#print(X)
print(Y)

reg= SVR(kernel='rbf')
reg.fit(X,Y.ravel())

pred=reg.predict([[10]])
pred=nrmY.inverse_transform(pred)
print(pred)

ypred=nrmY.inverse_transform(reg.predict(X))
print(r2_score(Y, ypred))

#plt.scatter(nrmX.inverse_transform(X),nrmY.inverse_transform(Y),color="green")
plt.plot(X,ypred,color="red")
plt.show()