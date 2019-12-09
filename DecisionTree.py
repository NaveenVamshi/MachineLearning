#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
df = pd.read_csv("salaries.csv")


# In[31]:


df.head()


# In[32]:


input = df.drop("salary_more_then_100k", axis="columns")


# In[33]:


target = df["salary_more_then_100k"]


# In[34]:


from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()


# In[35]:


input['company_n'] = le_company.fit_transform(input['company'])
input['job_n'] = le_company.fit_transform(input['job'])
input['degree_n'] = le_company.fit_transform(input['degree'])


# In[36]:


input


# In[41]:


input_n = input.drop(['company', 'job', 'degree'] ,  axis="columns")


# In[42]:


input_n.head()


# In[44]:


target.head()


# In[46]:


from sklearn import tree
model = tree.DecisionTreeClassifier()


# In[48]:


model.fit(input_n, target)


# In[49]:


# Is salary of Google, Computer Engineer, Bachelors degree > 100 k ?
model.predict([[2,1,0]])


# In[50]:


#Is salary of Google, Computer Engineer, Masters degree > 100 k ?
model.predict([[2,1,1]])


# In[ ]:




