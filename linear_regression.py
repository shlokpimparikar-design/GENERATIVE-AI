import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("placement.csv")

#first we are oberving the data's distribution
plt.scatter(df["cgpa"],df["package"])
plt.xlabel("cgpa")
plt.ylabel("package")
#plt.show()

#from graph we can see that data is making an approx linesr

#so we will do linear regression

#first cheak for null value

print(df.isnull().sum())  
#pandas .isnull() gonna gives us df of same table but true and false in stade of values
#(treats all TRUE as 1 and FALSE as 0 )
#sum on datafreme by defalut works on column so it will sum all column one buy one and is total sum is 0 than there is no null value


#find the independent and deoendent feature

x = df[["cgpa"]]  #x is cgpa column # basically we did slicing
y = df[["package"]] #y is package column 

#alternative method that professor used
# x = df.iloc[:,0] #0th cloumn (cgpa)
# y = df.iloc[:,1] #1st column (pkg)
# print(type(x),type(y))  --> x and y are pandas.series because of this in fit() we have to change x_train to 2d-array using x_test.value.reshape(-1:1)



#training and testing
#E.x: we have 200 row than
#80% for taing the mode --> (find weight and bias)
#20% for the testing --> cheak the performance of model after training

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 7027 ) #please look at class notes
#Since the original x was a 1D Series, the resulting x_train and x_test are also 1D Series.


from sklearn.linear_model import LinearRegression
lr = LinearRegression()    #we create class called lr

lr.fit(x_train,y_train)     #here x_train must be 2-d array or pandas.DataFrame #wiredly y can be 1-d or pandas.series
m = lr.coef_
b = lr.intercept_

#model training is done you can get package by pkg = m*(cgpa) + b


#model testing

y_pred = lr.predict(x_test) #<here class of y_pred will be 'numpy.ndarray'>
y_pred = pd.DataFrame(y_pred,columns=["y_pred"])  #changing y_pred in to dataframe of one column and also scence y_pred is 2-d array of only one element that's why only one column 
#print(type(y_pred),type(y_test))

#y_pred.sum()         #if you wanna print sum use y_pred.value.sum() if you don't that it will also print dtype

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"The R2 value of our model :{r2*100}%") 
