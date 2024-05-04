# Ex-05-Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the data file and import numpy, matplotlib and scipy.

2.Visulaize the data and define the sigmoid function, cost function and gradient descent.

3.Plot the decision boundary .

4.Calculate the y-prediction.
```

## Program:
```

/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: popuri sravani
RegisterNumber:  212223240117
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Placement_Data.csv")
print(dataset)

dataset = dataset.drop('sl_no',axis=1)

dataset =dataset.drop('salary',axis=1)

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
print(dataset.dtypes)

dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes

print(dataset)

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

print(Y)

theta = np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta, X, y, alpha, num_iterations):
    m=len(y)
    for i in range (num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)

def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5,1,0)
    return y_pred

y_pred = predict(theta, X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:
## DATASET
![Screenshot 2024-05-05 000808](https://github.com/sravanipopuri2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139778301/a31b6e4a-ea20-4c18-be90-8cd2e6b73f64)
## DATATYPES OF FEATURES
![Screenshot 2024-05-05 000819](https://github.com/sravanipopuri2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139778301/ac9945a6-b186-437e-9be2-9e872523e0f5)
## DATASET AFTER CONVERTING THE VALUES INTO CATEGORICAL VALUES
![Screenshot 2024-05-05 000828](https://github.com/sravanipopuri2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139778301/856c25fe-99d8-4905-9e61-8510594cdbe0)
## Y VALUES
![Screenshot 2024-05-05 000837](https://github.com/sravanipopuri2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139778301/f91ab468-0b0c-46bd-8cee-6102d915a99f)
## ACCURACY
![Screenshot 2024-05-05 000844](https://github.com/sravanipopuri2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139778301/0d6f15a1-e373-447d-a8b7-1f0d71737070)
## Y_PRED VALUES
![Screenshot 2024-05-05 000853](https://github.com/sravanipopuri2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139778301/568f9c05-abb6-48dc-904f-3b141803a700)
## Y VALUES
![Screenshot 2024-05-05 000902](https://github.com/sravanipopuri2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139778301/4674ee6c-77c7-4723-83b4-f7c5727bf68d)
## Y_PRED NEW VALUES WITH DIFFERENT INPUTS
![Screenshot 2024-05-05 000906](https://github.com/sravanipopuri2006/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139778301/4fdeeb87-a670-44c2-9444-5dcebb27f580)










## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

