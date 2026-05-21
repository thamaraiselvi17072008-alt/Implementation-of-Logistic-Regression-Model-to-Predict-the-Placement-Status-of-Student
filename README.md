# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:THAMARAISELVI.V
RegisterNumber: 212225040467
```
```
import numpy as np
X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
], dtype=float)
y = np.array([50, 45, 70, 80, 95], dtype=float)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std
X = np.c_[np.ones(X.shape[0]), X] 
n_features = X.shape[1]
weights = np.zeros(n_features)
learning_rate = 0.01
epochs = 1000
for epoch in range(epochs):
    for i in range(X.shape[0]):
        xi = X[i]
        yi = y[i]
        y_pred = np.dot(xi, weights)
        error = y_pred - yi
        weights -= learning_rate * error * xi
print("Trained Weights (including intercept):", weights)
y_pred_all = np.dot(X, weights)
print("Predicted values:", y_pred_all)
```
## Output:
<img width="1920" height="1080" alt="Screenshot 2026-05-21 211330" src="https://github.com/user-attachments/assets/cc1d01ca-8442-481c-a31e-8257011dc582" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
