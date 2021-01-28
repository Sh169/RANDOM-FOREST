#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


company= pd.read_csv("Company_Data.csv")
company.head()


# In[5]:


colnames = list(company.columns)


# In[6]:


company["Sales"].unique()


# In[8]:


company["Sales"].value_counts()


# In[10]:


np.median(company["Sales"])


# In[12]:


company["sales"]="<=7.49"


# In[14]:


company.loc[company["Sales"]>=7.49,"sales"]=">=7.49"


# In[16]:


company.drop(["Sales"],axis=1,inplace=True)


# In[18]:


##Encoding the data as model.fit doesnt convert string data to float
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_names in company.columns:
    if company[column_names].dtype == object:
        company[column_names]= le.fit_transform(company[column_names])
    else:
        pass


# In[20]:


##Splitting the data into input and output
featues = company.iloc[:,0:10]# i/p features
labels = company.iloc[:,10]# "sales" target variable


# In[21]:


##Splitting the data into TRAIN and TEST 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(featues,labels,test_size = 0.3,stratify = labels) 


# In[22]:


y_train.value_counts()


# In[23]:


y_test.value_counts()


# In[24]:


####MODEL BULIDING 

from sklearn.ensemble import RandomForestClassifier as RF

model =RF(n_jobs=4,n_estimators = 150, oob_score =True,criterion ='entropy') 
model.fit(x_train,y_train)# Fitting RandomForestClassifier model from sklearn.ensemble 
model.oob_score_


# In[25]:


model.estimators_ # 
model.classes_ # class labels (output)
model.n_classes_ # Number of levels in class labels -2
model.n_features_  # Number of input features in model 10 here.


# In[26]:


model.n_outputs_ # Number of outputs when fit performed


# In[27]:


model.predict(featues)


# In[28]:


##Predicting on training data
pred_train = model.predict(x_train)


# In[29]:


##Accuracy on training data
from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(y_train,pred_train)#1.0


# In[30]:


##Confusion matrix
from sklearn.metrics import confusion_matrix
con_train = confusion_matrix(y_train,pred_train)


# In[31]:


##Prediction on test data
pred_test = model.predict(x_test)


# In[32]:


##Accuracy on test data
accuracy_test = accuracy_score(y_test,pred_test)#0.8


# In[33]:


np.mean(y_test==pred_test)


# In[34]:


##Confusion matrix
con_test = confusion_matrix(y_test,pred_test)


# In[36]:


pip install six


# In[38]:


pip install graphviz


# In[39]:


pip install pydotplus


# In[40]:


import six
import sys
sys.modules['sklearn.externals.six'] = six
import joblib
sys.modules['sklearn.externals.joblib'] = joblib


# In[ ]:





# In[41]:


###### VISUALIZING THE ONE DECISION TREE IN RANDOM FOREST ####
from sklearn.tree import export_graphviz 
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image  


# In[68]:


conda install python-graphviz


# 
# ### 

# In[ ]:




