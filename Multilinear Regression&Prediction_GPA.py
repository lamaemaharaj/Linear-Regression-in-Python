#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import statsmodels.api as sm 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()


# In[10]:


raw_data = pd.read_csv('GPA&SAT_Scores.csv')


# In[11]:


raw_data


# In[12]:


data = raw_data 
data['Attendance']= data['Attendance'].map({'Yes':1,'No':0})


# In[13]:


data.describe()


# In[14]:


y = data["GPA"]
x1= data[['SAT','Attendance']]


# In[15]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# In[24]:


plt.scatter(data['SAT'],y)
yhat= 0.0017*data['SAT']+0.275
yhatno= 0.6439+0.0014*data['SAT']
yhatyes=0.8665+0.0014*data['SAT']
fig = plt.plot(data['SAT'], yhat, lw=3, color='black', label= 'Original Regression Line')
fig = plt.plot(data['SAT'],yhatno, lw=2, color='blue', label= 'No Attendance Reg Line')
fig = plt.plot(data['SAT'],yhatyes, lw=2, color='red', label= 'Yes Attendance Reg Line')
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.show()


# In[26]:


new_data = pd.DataFrame({'const':1,'SAT':[1700,1670],'Attendance':[0,1]})
new_data = new_data[['const','SAT','Attendance']]
new_data


# In[27]:


new_data.rename(index={0:'Bob',1:'Alice'})


# In[29]:


prediction= results.predict(new_data)
prediction


# In[33]:


predictiondf=pd.DataFrame({'Predictions': prediction})
joined = new_data.join(predictiondf)
joined.rename(index={0:'Bob',1:'Alice'})


# In[ ]:




