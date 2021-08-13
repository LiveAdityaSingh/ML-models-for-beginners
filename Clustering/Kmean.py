# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 19:27:05 2020

@author: Aditya
"""

#kmeans 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("./Mall_Customers.csv")

x=dataset.iloc[:,3].values
x=x.reshape(-1,1)
Xo=dataset.iloc[:,[3,4]].values
y=dataset.iloc[:,4].values

plt.scatter(x,y)
plt.xlabel('Score')
plt.ylabel('Salary')
plt.show()

from sklearn.cluster import KMeans

wcss=[]
for i in range(1,10):
    km=KMeans(n_clusters=i)
    km.fit(Xo)
    wcss.append(km.inertia_)
    
plt.plot(range(1,10),wcss)
plt.xlabel('K')
plt.ylabel("wcss")
plt.show()

km=KMeans(n_clusters=5)
ym=km.fit_predict(Xo)
print(ym)

plt.scatter(Xo[ym==0,0],Xo[ym==0,1],s=5,color="green",label="one")
plt.scatter(Xo[ym==1,0],Xo[ym==1,1],s=5,color="red",label="two")
plt.scatter(Xo[ym==2,0],Xo[ym==2,1],s=5,color="grey",label="three")
plt.scatter(Xo[ym==3,0],Xo[ym==3,1],s=5,color="blue",label="four")
plt.scatter(Xo[ym==4,0],Xo[ym==4,1],s=5,color="yellow",label="five")
plt.legend()
plt.show()
