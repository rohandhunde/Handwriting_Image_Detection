#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import the important libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as  np
import pylab as py
from tensorflow.keras.layers import Dense,LSTM


# In[3]:


# import the prebuild dataset from tensorflow
# Split the data into train and test set
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()


# In[4]:


# lets vissuallize the sample image from the X_train set
plt.matshow(x_train[0])


# In[5]:


# Check the shape of the train dataset that we already splitted
x_train[0].shape,y_train.shape[0]


# In[6]:


# Reshape the size of image of the X_train and X_test
x_train=x_train/255.0
x_test=x_test/255.0
y_train[:5]


# In[7]:


# Create the Deep learning model
model = models.Sequential()
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(LSTM(128,activation="relu"))
model.add(Dense(1000,activation="relu"))
model.add(Dense(10,activation="softmax"))


# In[10]:


#Lets Compile the Model
model.compile(optimizer="adam",
             loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])


# In[11]:


# Fit the model
model.fit(x_train,y_train,epochs=10)


# In[12]:


# lets predict the X_Test
predictions=model.predict(x_test)
np.argmax(predictions[5])


# In[13]:


# lets check the answer from the above ouput
plt.matshow(x_test[5])


# # Check the predictions 1 so what it will be returns 

# In[18]:


# lets check the another examples
np.argmax(predictions[1])


# In[22]:


# model predicts the 2 lets check what it will gives us
plt.imshow(x_test[1])

