# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import Libraries: pandas,numpy,matplotlib,sklearn.
2. Load Dataset: Read CSV file containing study hours and marks.
3. Check Data: Preview data and check for missing values.
4. Define Variables: Set x=Hours, y=Scores.
5. Split Data: Train-test spilt(80-20).
6. Train Model: Fit Linear Regression on training data.
7. Predict: use model to predict scores on test data.
8. Evaluate: Calculate Mean Absolute Error(MAE) and R^2 score.
9. Visualize:Plot actual data and regression line.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: R.Rubasri
RegisterNumber:  212224240139
*/
```
```
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('student_scores.csv')
print(df.head())
print(df.tail())

# Independent and dependent variables
X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values
print("X:", X)
print("Y:", Y)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Train model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict
Y_pred = regressor.predict(X_test)
print("Predicted:", Y_pred)
print("Actual:", Y_test)

# Training set visualization
plt.scatter(X_train, Y_train, color="orange")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Testing set visualization
plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, regressor.predict(X_test), color="green")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Error metrics
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
```
<br>
<br>
<br>

## Output:
### Head Values:
<img width="178" height="91" alt="image" src="https://github.com/user-attachments/assets/95a5b193-31a8-4238-b11b-57f59e6a7024" />

### Tail Values:
<img width="246" height="91" alt="image" src="https://github.com/user-attachments/assets/5cf50404-281f-4085-9034-160af44b5602" />

### Compare Dataset:
<img width="509" height="408" alt="image" src="https://github.com/user-attachments/assets/f7b96cf5-e097-43c2-9b19-d25d7e8afe95" />

### Predication values of X and Y:
<img width="543" height="43" alt="image" src="https://github.com/user-attachments/assets/15fb823b-0063-4fd0-9600-a0dd857e7afd" />

### Training set:
<img width="569" height="400" alt="image" src="https://github.com/user-attachments/assets/b9bbac8a-cea4-429c-ac2b-f1acc96d5393" />

### Testing Set:
<img width="630" height="400" alt="image" src="https://github.com/user-attachments/assets/146e5c4e-44b0-44be-9c3b-14846b873893" />

### MSE,MAE and RMSE:
<img width="298" height="45" alt="image" src="https://github.com/user-attachments/assets/6b568cc9-2346-4219-8408-ee7fb208e7c5" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
