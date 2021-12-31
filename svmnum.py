import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
data=pd.read_csv('E:\\heart.csv')
cols=data.shape[1]
x=data.iloc[ : ,0:cols-1].values
y=data.iloc[ : ,cols-1:cols].values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
from sklearn.svm import SVC
model = SVC(kernel = 'linear', random_state = 0)
model.fit(X_train, y_train.ravel())
y_pre=model.decision_function(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score,plot_confusion_matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
