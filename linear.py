import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

data = pd.read_csv(r'C:\Users\ibrahima\Desktop\auto.txt')
data.head()
data = data[["displacement","horsepower"]]
plt.scatter(data["displacement"],data["horsepower"],color="blue")
plt.xlabel("displacement")
plt.ylabel("horsepower")
plt.show()

train = data[:(int(len(data)*0.8))]
test = data[(int(len(data)*0.8)):]

train_x = np.array(train[["displacement"]])
train_y = np.array(train[["horsepower"]])

regr = linear_model.LinearRegression()
regr.fit(train_x,train_y)

print("coefficient : ", regr.coef_)
print("intercept  : ", regr.intercept_)

plt.scatter(train["displacement"],train["horsepower"],color="blue")
plt.plot(train_x,train_x * regr.coef_ + regr.intercept_, '-r')
plt.show()

test_x = np.array(test[["displacement"]])
test_y = np.array(test[["horsepower"]])
test_y_= regr.predict(test_x)

print(" mean absolute error  %.2f " % np.mean(np.absolute(test_y_ - test_y)))
print(" R2score  %.2f " % r2_score(test_y_ , test_y))

