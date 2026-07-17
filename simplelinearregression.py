import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('/Users/nitinkumar/Desktop/april(2025)/21st- SLR/21st- SLR/SIMPLE LINEAR REGRESSION/Salary_Data.csv')

x= dataset.iloc[:, :-1]
y= dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test ,y_train, y_test =train_test_split(x,y,test_size=0.20,random_state=0)


from sklearn.linear_model import LinearRegression
regressor =LinearRegression()
regressor.fit(x_train,y_train)

y_predict = regressor.predict(x_test)

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title('salary vs experiences')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

m_slope= regressor.coef_
print(m_slope )

c_intercept = regressor.intercept_
print(c_intercept)


bias = regressor.score(x_train,y_train)
print(bias) 

variance= regressor.score(x_test, y_test)
print(variance)


y_mean = np.mean(y)
ssr =np.sum((y_predict-y_mean)**2)
print(ssr)



y = y[0:6]
print(y)
sse = np.sum((y-y_predict)**2)
print(sse)

mean_total = np.mean(dataset.values)
sst = np.sum((dataset.values-mean_total)**2)
print(sst)



r_square = 1-ssr/sst
print(r_square)


import pickle

filename ='lineaer_regression_model.pkl'

with open(filename,'wb') as file:
    pickle.dump(regressor,file)
    
print("model has been picked and saved as linear_regression_model")
    
import os

os.getcwd()
    

