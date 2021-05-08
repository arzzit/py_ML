#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


# In[2]:


digits = load_digits()


# In[3]:


digits.data[0]


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


(X_train,X_test,Y_train,Y_test)=train_test_split(digits.data,digits.target,test_size = 0.2)


# In[13]:


from sklearn.linear_model import LogisticRegression


# In[14]:


model = LogisticRegression()


# In[15]:


model.fit(X_train,Y_train)


# In[16]:


model.score(X_test,Y_test)


# In[17]:


y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix


# In[18]:


cm = confusion_matrix(y_predicted,Y_test)


# In[19]:


cm


# In[21]:



import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




