"""
Created on Thu Jun  4 19:23:29 2020

@author: Aditya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.metrics import r2_score

dataset=pd.read_csv("./Position_Salary.csv")
X=dataset.iloc[:,1].values        #input
Y=dataset.iloc[:,2].values          #output

X=X.reshape(-1,1)
#Y=Y.reshape(-1,1)
plt.scatter(X,Y,color="blue")
print(X)
print(Y)

reg=RF(n_estimators=200,max_leaf_nodes=9,random_state=1)
reg.fit(X,Y)

print(reg.estimator_[5].predict([[7.5]]))

grid=np.arange(min(X),max(X),0.1)
grid=grid.reshape(-1,1)
plt.plot(grid,reg.predict(grid),color="green")
#plt.plot(X,Y,color="red")
plt.show()

print(r2_score(Y, reg.predict(X)))
