import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model



data = pd.read_csv(r'C:\Users\ibrahima\Desktop\auto.txt')
data.head()

X = data[["cylinders","mpg","displacement","weight","acceleration"]]
Y = data[["horsepower"]]


# Generating training and testing data from our data:
# We are using 80% data for training.
train = data[:(int((len(data)*0.8)))]
test = data[(int((len(data)*0.8))):]

#Modeling:
#Using sklearn package to model data :
regr = linear_model.LinearRegression()

train_x = np.array(train[["cylinders","mpg","displacement","weight","acceleration"]])
train_y = np.array(train["horsepower"])

regr.fit(train_x,train_y)

test_x = np.array(test[["cylinders","mpg","displacement","weight","acceleration"]])
test_y = np.array(test["horsepower"])


# print the coefficient values:
coeff_data = pd.DataFrame(regr.coef_ , X.columns , columns=["Coefficients"])

print(coeff_data)


#Now let’s do prediction of data:
Y_pred = regr.predict(test_x)
# Check accuracy:
from sklearn.metrics import r2_score
R = r2_score(test_y , Y_pred)
print ("R² :",R)

