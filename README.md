# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess data – Read dataset, convert labels (Placed = 1, Not Placed = 0), and select features.

2. Normalize features – Apply standard scaling to bring all inputs to the same range.

3. Initialize model – Add bias term, define sigmoid function, and initialize weights (θ).

4. Train using Gradient Descent – Update weights iteratively by minimizing the cost function.

5. Predict and evaluate – Generate predictions using sigmoid and calculate accuracy. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VIKASH.S
RegisterNumber:  212225040490
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("Placement_Data (1).csv")
data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})
X = data[["ssc_p", "mba_p"]].values
y = data["status"].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
m = len(y)
X = np.c_[np.ones(m), X]
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []
for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1 / m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    cost = cost_function(X, y, theta)
    cost_history.append(cost)
y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)
accuracy = np.mean(y_pred == y) * 100
print("Weights:", theta)
print("Accuracy:", accuracy, "%")
plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Logistic Regression using Gradient Descent")
plt.show()
```

## Output:
<img width="986" height="629" alt="image" src="https://github.com/user-attachments/assets/e0ab4e91-aa06-4a78-84bb-33137ad1a812" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

