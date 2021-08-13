# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:19:45 2020

@author: Aditya
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor as dt
from sklearn.tree import export_graphviz

dataset=pd.read_csv("./Position_Salary.csv")
X=dataset.iloc[:,1].values        #input
Y=dataset.iloc[:,2].values          #output

X=X.reshape(-1,1)
Y=Y.reshape(-1,1)
plt.scatter(X,Y,color="blue")

reg=dt(max_leaf_nodes=5)
reg.fit(X,Y)

print(reg.predict([[10.5]]))

print(reg.predict([[8.5]]))

print(reg.predict([[8.4]]))


grid=np.arange(min(X),max(X),0.1)
grid=grid.reshape(-1,1)
plt.plot(grid,reg.predict(grid),color="green")
plt.plot(X,Y,color="red")
plt.show()

export_graphviz(reg, out_file="./tree.doc", max_depth=2)


