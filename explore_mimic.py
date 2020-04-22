import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from mimicloader import *

class confusion_matrix_metrics(object):
    def __init__(self,M):
        self.M = M
        self.TP = M[0,0]
        self.FN = M[0,1]
        self.FP = M[1,0]
        self.TN = M[1,1]

    def sensitivity(self):
        return self.TP/(self.TP + self.FN)
    def specificity(self):
        return self.TN/(self.TN + self.FP)
    def precision(self):
        return self.TP/(self.TP + self.FP)
    def negative_predictive_value(self):
        return self.TN/(self.TN + self.FN)
    def miss_rate(self):
        return 1 - self.sensitivity()
    def fall_out(self):
        return 1 - self.specificity()
    def false_discovery(self):
        return 1 - self.precision()
    def false_omission(self):
        return 1 - self.negative_predictive_value()
    def threat_score(self):
        return self.TP/(self.TP + self.FN + self.FP)
    def accuracy(self):
        return (self.TP + self.TN)/(self.TP + self. TN + self.FP + self.FN)
    def balanced_accuracy(self):
        return self.sensitivity()/2 + self.specificity()/2
    def F1(self):
        return 2*self.sensitivity()*self.precision()/(self.sensitivity()+self.precision())

filepath = 'mimic_dataset.csv'
loader = MIMICLoader()
df = loader.load(filepath)
df1 = df[df.hospital_expire_flag == 1]
df2 = df[df.hospital_expire_flag == 0]
df2 = df2.sample(n=len(df1))
df3 = df1.append(df2)
y = df3.hospital_expire_flag
X = df3.drop(columns=['hospital_expire_flag'])

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=42)

clf = AdaBoostClassifier(n_estimators=1000,learning_rate=0.5)
clf.fit(X_train,y_train)
yhat_ada = clf.predict(X_test)
clf.score(X_test,y_test)

clf = RandomForestClassifier(n_estimators=1000,criterion='gini')
clf.fit(X_train,y_train)
yhat_forest = clf.predict(X_test)
clf.score(X_test,y_test)

M_for = confusion_matrix(yhat_forest, y_test)
M_ada = confusion_matrix(yhat_ada, y_test)

metric_ada = confusion_matrix_metrics(M_ada)
metric_forest = confusion_matrix_metrics(M_for)
