#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


df = pd.read_csv("insurance_data.csv")
df.head()


# In[40]:


plt.xlabel('age')
plt.ylabel("insurance")
plt.scatter(df.age, df.bought_insurance,color="green", marker="*")


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


X_train, X_test , Y_train , Y_test= train_test_split(df[["age"]], df.bought_insurance , train_size= 0.9)


# In[43]:


X_test


# In[44]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[45]:


model.fit(X_train, Y_train)


# In[46]:


Y_predict = model.predict(X_test)


# In[47]:


model.score(X_test, Y_test)


# In[52]:


model.predict_proba(X_test)


# In[50]:


model.score(X_test,Y_test)


# In[62]:


model.predict([[22]])


# In[ ]:




