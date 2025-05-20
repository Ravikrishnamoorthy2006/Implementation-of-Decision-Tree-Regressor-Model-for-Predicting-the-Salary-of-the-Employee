# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the standard libraries.

2. Upload the dataset and check for any null values using .isnull() function.

3. Import LabelEncoder and encode the dataset.

4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5. Predict the values of arrays.

6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7. Predict the values of array.

8. Apply to new unknown values. 

## Program:

/*

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: Ravikrishnamoorthy D

RegisterNumber: 212224040271

*/
```
import pandas as pd
data = pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position", "Level"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
mse

r2 = metrics.r2_score(y_test, y_pred)
r2

dt.predict([[5,6]])
```

## Output:

![image](https://github.com/user-attachments/assets/da0abb5f-bbc7-4146-a1f4-44c12ce13369)

![image](https://github.com/user-attachments/assets/17c5507a-6dfa-4d14-942d-e0691e3dae1e)

![image](https://github.com/user-attachments/assets/8decafa0-20ee-4a86-a6ef-a655ba0d7947)

![image](https://github.com/user-attachments/assets/3490d4aa-3288-47cb-bf02-83cb39a5401d)

![image](https://github.com/user-attachments/assets/62c34b02-4cda-4e21-9e97-a0340404bbf2)

![image](https://github.com/user-attachments/assets/f2f18783-0b0b-4ddd-82f8-15052710e37c)

![image](https://github.com/user-attachments/assets/c8890993-1832-443f-bc0e-b080de770bad)











## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
