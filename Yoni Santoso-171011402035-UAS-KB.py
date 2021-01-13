#!/usr/bin/env python
# coding: utf-8

# In[95]:


# Yoni Santoso
# 171011402035
# 07 TPLE 007
# UAS Kecerdasan Buatan

import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import math
import sklearn

#Menampilkan Hours dan Scores

general_data= pd.read_csv("student scores.csv")
general_data.head(26)


# In[53]:


general_data.shape


# In[92]:


#Menampilkan Jumlah Data

print("#JUMLAH DATASET SAYA =" +str(len(general_data.index)))


# In[91]:


#Menampilkan Deskriptif Analisis

general_data.describe()


# In[90]:


#visualisasi hasil plot hours vs scores

general_data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[57]:


x = general_data.iloc[:, :-1].values
y = general_data.iloc[:, :1].values


# In[58]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[80]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x_train, y_train)


# In[61]:


print(regressor.intercept_)


# In[62]:


print(regressor.coef_)


# In[64]:


y_pred = regressor.predict(x_test)


# In[89]:


#visualisasi hours vs scores

sns.pairplot(general_data)


# In[88]:


#visualisasi hasil training set

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Hours vs Scores')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.show()


# In[87]:


#visualisasi hasil test set

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Hours vs Scores')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.show()


# In[ ]:




