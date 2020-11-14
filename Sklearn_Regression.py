#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression


# In[48]:


data = pd.read_csv('1.01.+Simple+linear+regression.csv')
data.head()


# In[49]:


# Variable Declaration
x = data['SAT']
y = data['GPA']


# In[65]:


x_matrix = x.values.reshape(84,1)
y_matrix = y.values.reshape(84,1)


# In[68]:


reg = LinearRegression()
reg.fit(x_matrix,y_matrix)


# In[69]:


# R_Squared 
reg.score(x_matrix,y_matrix)


# In[70]:


# Coeffictients 
reg.coef_


# In[71]:


# Intercept 
reg.intercept_


# In[80]:


# Predictions 
reg.predict([[1740]])


# In[87]:


# Variables for GPA Prediction
new_data['Predicted_GPA'] = pd.DataFrame(data=[1740,1760])
new_data


# In[90]:


# Data Model & Prediction Plot
plt.scatter(x,y)
yhat = reg.coef_*x_matrix+reg.intercept_
fig = plt.plot(x,yhat,lw=4,color='red', label='Regression Line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()


# In[ ]:




