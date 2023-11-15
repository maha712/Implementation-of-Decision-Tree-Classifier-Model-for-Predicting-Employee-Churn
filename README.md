# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by:mahalakshmi.k

RegisterNumber:212222240057  
*/

import pandas as pd

data=pd.read_csv("/content/Employee.csv")


data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])

data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()

y=data["left"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)


from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion="entropy")

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)


from sklearn import metrics

accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])


## Output:

Initial data set

![initial data set](https://github.com/maha712/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121156360/2d994fa8-37ea-488e-82b2-2b7f1ecc27ea)

Data info

![data info](https://github.com/maha712/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121156360/497c35ee-ce6b-4a5f-9da4-3ce517f39fe4)

Optimization of null values

![optimization of null values](https://github.com/maha712/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121156360/5df3e903-f6ec-464d-a257-91b5c53ab434)

Assignment of X and Y values

![assignment of X and Y values](https://github.com/maha712/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121156360/3b740feb-9776-4b61-a95d-a5f9ced0b40f)

![assign](https://github.com/maha712/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121156360/0c377814-9914-4d24-a324-f0d5df6c3ce0)

Converting string literals to numerical values using label encoder

![Converting string literals to numerical values using label encoder](https://github.com/maha712/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121156360/8acd578b-888a-4f5f-9689-4d9af741108d)

Accuracy

![Accuracy](https://github.com/maha712/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121156360/da0f8689-abbd-42ae-96de-ea698fcc1a53)

Prediction
![Prediction](https://github.com/maha712/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121156360/8a16d4f4-36cb-4751-a874-383294514750)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
