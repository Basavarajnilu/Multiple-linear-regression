#importing the tools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix

#loading the data
train=pd.read_csv("file:///C:/Users/HP/Desktop/train (1).csv")
test=pd.read_csv("file:///C:/Users/HP/Desktop/test (2).csv")

#EDA
train = train.dropna()
test = test.dropna()
train.head()
test.head()

#building the model
X_train = np.array(train.iloc[:, :-1].values)
y_train = np.array(train.iloc[:, 1].values)
X_test = np.array(test.iloc[:, :-1].values)
y_test = np.array(test.iloc[:, 1].values)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)

#ploting the graph
plt.plot(X_train, model.predict(X_train), color='green')
plt.show()
print(accuracy)
