# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:14:40 2020

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset_tain=pd.read_csv('train.csv')
Sex = {'male': 1,'female': 0}
dataset_tain.Sex=[Sex[item] for item in dataset_tain.Sex]
dataset_tain.drop(["Name"],axis=1,inplace=True)
dataset_tain.drop(["Ticket"],axis=1,inplace=True)
dataset_tain.drop(["Cabin"],axis=1,inplace=True)
dataset_tain["Embarked"].fillna("L",inplace=True)
Embarked={'S':0,'C':1,'Q':2,'L':3}

dataset_tain.Embarked=[Embarked[item] for item in dataset_tain.Embarked]
dataset_tain.fillna(dataset_tain.mean())


dataset_tain=dataset_tain.fillna(dataset_tain.mean())
X_train=dataset_tain.iloc[:,:-1].values
y_train=dataset_tain.iloc[:,8].values
dataset_test=pd.read_csv('test.csv')
Sex = {'male': 1,'female': 0}
dataset_test.Sex=[Sex[item] for item in dataset_test.Sex]
dataset_test.drop(["Name"],axis=1,inplace=True)
dataset_test.drop(["Ticket"],axis=1,inplace=True)
dataset_test.drop(["Cabin"],axis=1,inplace=True)    
dataset_test["Embarked"].fillna("L",inplace=True)
Embarked={'S':0,'C':1,'Q':2,'L':3}


dataset_test.Embarked=[Embarked[item] for item in dataset_test.Embarked]
dataset_test=dataset_test.fillna(dataset_test.mean())
X_test=dataset_test.iloc[:,:-1].values
y_test=dataset_test.iloc[:,8].values



#fitting the classifier 
from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)