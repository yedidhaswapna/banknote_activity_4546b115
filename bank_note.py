import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

Bank = pd.read_csv("BankNoteML/BankNote_Authentication.csv")

X=Bank.iloc[:,:-1]#until last column
y=Bank.iloc[:, -1]#Last column

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

classifier = DecisionTreeClassifier()

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

print(y_pred)

score = accuracy_score(y_test,y_pred)
score

pickle_out = open("BankNoteML.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()

dtc.predict([[0,1,2,3]])

