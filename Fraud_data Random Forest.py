#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot


# In[4]:


fraud_data = pd.read_csv("Fraud_check.csv")
fraud_data.head()


# In[3]:


fraud_data.columns


# In[5]:


##Converting the Taxable income variable to bucketing. 

fraud_data["income"]="<=30000"


# In[6]:


fraud_data.loc[fraud_data["Taxable.Income"]>=30000,"income"]="Good"
fraud_data.loc[fraud_data["Taxable.Income"]<=30000,"income"]="Risky"


# In[7]:


##Droping the Taxable income variable
fraud_data.drop(["Taxable.Income"],axis=1,inplace=True)


# In[8]:


#to reduce the complexity lets change the columns names
fraud_data.rename(columns={"Undergrad":"undergrad","Marital.Status":"marital","City.Population":"population","Work.Experience":"experience","Urban":"urban"},inplace=True)


# In[9]:


## Model doesnt not consider String ,So lets label the categorical columns

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in fraud_data.columns:
    if fraud_data[column_name].dtype == object:
        fraud_data[column_name] = le.fit_transform(fraud_data[column_name])
    else:
        pass


# In[10]:


##Splitting the data into i/p and o/p
features = fraud_data.iloc[:,0:5]
labels = fraud_data.iloc[:,5]


# In[11]:


## Collecting the column names

colnames = list(fraud_data.columns)
predictors = colnames[0:5]#feature variable 
target = colnames[5]# targated variable


# In[12]:


################# Splitting the data into TRAIN and TEST ##########################

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)


# In[13]:


##Model Building


# In[14]:


from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)


# In[15]:


model.estimators_
model.classes_
model.n_features_
model.n_classes_


# In[16]:


model.n_outputs_

model.oob_score_ 


# In[17]:


##Predictions on train data
prediction = model.predict(x_train)


# In[18]:


# For accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)


# In[19]:


np.mean(prediction == y_train)


# In[20]:


##Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)


# In[21]:


##Prediction on test data
pred_test = model.predict(x_test)


# In[22]:


##Accuracy
acc_test =accuracy_score(y_test,pred_test)


# In[ ]:





# In[ ]:




