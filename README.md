# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### Step 1: Start
#### Step 2: Import Libraries
#### Step 3: Load the Dataset
#### Step 4: Split the Data
#### Step 5: Instantiate the Model
#### Step 6: Train the Model
#### Step 7: Make Predictions
#### Step 8: Evaluate the Model
#### Step 9: Stop

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mohamed Athif Rahuman J
RegisterNumber: 212223220058
```
```
import pandas as pd
data = pd.read_csv("D:/introduction to ML/jupyter notebooks/sample/Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data.head()
data.isnull()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")# library for large linear classificiation
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85,]])
```

## Output:
### Preprocessing:
![image](https://github.com/arbasil05/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144218037/05c1df2e-f3f4-4b47-aba5-1c37bd368831)
### Classifictaion: 
![image](https://github.com/arbasil05/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144218037/30780441-5caa-496d-8868-2e4f30040078)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
