#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix


# In[16]:


train=pd.read_csv("file:///C:/Users/HP/Desktop/train (1).csv")
test=pd.read_csv("file:///C:/Users/HP/Desktop/test (2).csv")


# In[19]:


train = train.dropna()
test = test.dropna()
train.head()


# In[20]:


train.info()


# In[21]:


test.head()


# In[9]:


test.info()


# In[22]:


np.mean(train)


# In[23]:


np.mean(test)


# In[25]:




X_train = np.array(train.iloc[:, :-1].values)
y_train = np.array(train.iloc[:, 1].values)
X_test = np.array(test.iloc[:, :-1].values)
y_test = np.array(test.iloc[:, 1].values)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)

plt.plot(X_train, model.predict(X_train), color='green')
plt.show()
print(accuracy)


# In[28]:


from sklearn import datasets, linear_model, metrics 
  
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train) print('Coefficients: \n', reg.coef_) 
  
print('Variance score: {}'.format(reg.score(X_test, y_test))) 


# In[29]:


print('Coefficients: \n', reg.coef_) 
  


# In[ ]:




