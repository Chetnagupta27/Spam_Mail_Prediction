#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# TfidfVectorizer is used to convert text values to numerical values.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Data Collection and Preprocessing

# In[4]:


# loading the data into pandas dataframe
raw_mail_data=pd.read_csv('mail_data.csv')


# In[5]:


print(raw_mail_data)


# In[6]:


#replace the null values with a null string
mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')


# In[7]:


#printing first five rows of dataframe
mail_data.head()


# In[8]:


mail_data.tail()
#printing last 5 rows of data


# In[9]:


mail_data.shape
# Here 5572--> Mails
# HEre 2 --> Spam or Ham i.e. Label


# Label Encoding

# In[10]:


#label spam mails as 0 and ham mails as 1
mail_data.loc[mail_data['Category']=='spam','Category',]=0
mail_data.loc[mail_data['Category']=='ham','Category',]=1


# spam = 0
# ham = 1

# In[11]:


# Seprating the data as text and label
x=mail_data['Message']
y=mail_data['Category']


# In[12]:


print(y)


# In[13]:


print(x)


# Train Test Split of Data

# In[14]:


x_train , x_test , y_train , y_test=train_test_split(x,y,test_size=0.2 , random_state=3)


# In[17]:


print(x.shape)


# In[18]:


print(y.shape)


# Feature Extraction

# In[21]:


# Transform the text data to features that can be used as imput to the Logistic Regression
feature_extraction= TfidfVectorizer(min_df=1,stop_words='english', lowercase='True')
x_train_features=feature_extraction.fit_transform(x_train)
x_test_features=feature_extraction.transform(x_test)

# convert y_train and y_test values as integers
y_train=y_train.astype('int')
y_test=y_test.astype('int')


# In[22]:


print(x_train_features)


# In[24]:


print(x_test_features)


# In[25]:


print(y_train)


# In[26]:


print(y_test)


# Training the Logistic Regression Model

# In[27]:


model=LogisticRegression()


# In[28]:


# training the logistic regression model with the training data
model.fit(x_train_features, y_train)


# Evaluating the trained model

# In[29]:


# prediction on training data
prediction_on_train=model.predict(x_train_features)
accuracy_on_train=accuracy_score(y_train, prediction_on_train)


# In[30]:


print("Accuracy on training data : ", accuracy_on_train)


# In[31]:


# prediction on test data
prediction_on_test=model.predict(x_test_features)
accuracy_on_test=accuracy_score(y_test, prediction_on_test)


# In[32]:


print("Accuracy score : ", accuracy_on_test )


# Building a predictive System

# In[35]:


input_mail=["I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."]

# convert text to feature vectors
input_data_features=feature_extraction.transform(input_mail)
            
# making predictions
prediction= model.predict(input_data_features)
print(prediction)

if(prediction[0]==1):
    print("Ham Mail")
else:
    print("Spam Mail")


# In[ ]:




