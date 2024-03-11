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
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: mohamed athif rahuman J
RegisterNumber:  212223220058
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("Placement_Data.csv")
df
df.head()
df.tail()
df=df.drop(['sl_no','gender','salary'],axis=1)
df=df.drop(['ssc_b','hsc_b'],axis=1)
df.shape
df.info()
df["degree_t"]=df["degree_t"].astype("category")
df["hsc_s"]=df["hsc_s"].astype("category")
df["workex"]=df["workex"].astype("category")
df["status"]=df["status"].astype("category")
df["specialisation"]=df["specialisation"].astype("category")
df["degree_t"]=df["degree_t"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df["workex"]=df["workex"].cat.codes
df["status"]=df["status"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
x=df.iloc[: ,:-1].values
y=df.iloc[:,- 1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

df.head()
from sklearn.linear_model import LogisticRegression

#printing its accuracy
clf=LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion 

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
# Predicting for random value
clf.predict([[1	,78.33,	1,	2,	77.48,	2,	86.5,	0,	66.28]])

```

## Output:
Dataset
![image](https://github.com/mdathif12/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365313/41c53bc5-f701-4e06-8b35-fa519f737723)

![image](https://github.com/mdathif12/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365313/757956a7-260e-4df0-adbc-a11d5f242a88)

![image](https://github.com/mdathif12/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365313/e9141d95-2141-4035-8e94-040b4d54849f)

![image](https://github.com/mdathif12/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365313/f7eab280-bcca-409b-93b1-cf8294e8d78b)

![image](https://github.com/mdathif12/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365313/61e075ba-02a6-49eb-a98a-d5e7da1bc6e9)

![image](https://github.com/mdathif12/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365313/80e8c8c4-d232-46da-bd50-6597272695d6)

![image](https://github.com/mdathif12/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365313/1c0995c9-ecad-40c7-848f-855adab0c0ea)

![image](https://github.com/mdathif12/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365313/378a6b64-00ce-484e-afce-7153042e2f54)

![image](https://github.com/mdathif12/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365313/5faf87b5-cdd5-4b15-9a6d-00ff5ee19484)

![image](https://github.com/mdathif12/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365313/d7517006-e147-4dbf-8abb-081e38d08ac6)

![image](https://github.com/mdathif12/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365313/7afb4e55-1d30-4ca7-adb8-9a6f05b09058)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
