#!/usr/bin/env python
# coding: utf-8

# In[2]:


# y = m1x +m2x+ m3x + c
import pandas as pd
import numpy as np
from sklearn import linear_model


# In[7]:


# y = m1x +m2x+ m3x + c
df = pd.read_csv("homeprices.csv")
df


# In[13]:


# This is done in the case value has some decimal value but doing generally for normally
import math
d= math.floor(df.bedrooms.median())
d


# In[22]:


df.bedrooms= df.bedrooms.fillna(d)
df


# In[25]:


reg = linear_model.LinearRegression()
reg.fit(df[["area", "bedrooms" , "age"]], df.price)


# In[26]:


reg.coef_


# In[27]:


reg.intercept_


# In[29]:


reg.predict([[3000, 3, 40]])


# In[31]:


http://localhost:8888/notebooks/Naveen_codebasic/MultiReg.ipynb##rechecking  y = m1x +m2x+ m3x + c y=price
3000*112.06244194+ 3*23388.88007794+ (40*-3231.71790863)+ 221323.00186540408


# In[ ]:




