import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


df = pd.read_csv("placement.csv")


plt.scatter(df["cgpa"],df["package"])
plt.xlabel("cgpa")
plt.ylabel("package")


#print(df.isnull().sum())  



x = df[["cgpa"]]  
y = df[["package"]]  
best_seed = [0,0]

for i in range(10000):

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = i ) 


    lr = LinearRegression()    

    lr.fit(x_train,y_train)    
    m = lr.coef_
    b = lr.intercept_


    y_pred = lr.predict(x_test) 
    y_pred = pd.DataFrame(y_pred,columns=["y_pred"])


    r2 = r2_score(y_test, y_pred)
    if best_seed[1] < r2:
        best_seed[1] = r2
        best_seed[0] = i
    
print(f"Best seed is :{best_seed[0]} and R2 for that is {best_seed[1]}")
print("NOTE that finding best seed is not standard prectice because seed changes based on given data to ml")
print("This seed is best only for data in placement.csv")
