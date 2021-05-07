#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
from word2number import w2n
from sklearn import linear_model


# In[2]:


df=pd.read_csv(r'D:\Hiring.csv')
df


# In[3]:


df.experience=df.experience.fillna("zero")
df


# In[4]:


df.experience=df.experience.apply(w2n.word_to_num)
df


# In[8]:


df.test_score(out of 10).median()


# In[6]:


df.experience.median()


# In[10]:


df['test_score(out of 10)'].fillna(10)


# In[12]:


df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(10)


# In[13]:


df


# In[16]:


reg=linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])


# In[18]:


reg.predict([[200,5,9]])


# In[19]:


reg.predict([[2,9,6]])


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(['test_score(out of 10)'],['interview_score(out of 10)'],['salary($)'],df.experience)
plt.title('4 VARIABLE')                                                      
plt.xlabel('Yes')
plt.ylabel('No')
plt.show()                                                     


# In[ ]:




