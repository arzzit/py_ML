#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import linear_model


# In[2]:


df = pd.read_csv(r'D:\Homeprices.csv')
df


# In[3]:


df=df.drop('Unnamed: 4',axis=1)
df


# In[4]:


df.bedrooms.median()


# In[5]:


df.bedrooms=df.bedrooms.fillna(df.bedrooms.median())
df


# In[9]:


reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)


# In[10]:


reg.coef_


# In[11]:


reg.intercept_


# In[12]:


reg.predict([[5000,2,50]])


# In[13]:


reg.predict([[3000,4,15]])


# In[ ]:
with open('model_pickle','wb') as f:
    pickle.dump(reg,f)

with open('model_pickle','rb') as f:
    mp=pickle.load(f)
print(mp)

