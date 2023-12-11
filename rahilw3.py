#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


get_ipython().system('pip install keras')


# In[3]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[6]:


for i in range(9):
	# define subplot
	plt.subplot(330 + 1 + i)
	# plot raw pixel data
	plt.imshow(train_images[i])


# In[7]:


train_images, test_images = train_images / 255.0, test_images / 255.0


# In[8]:


train_labels = tf.one_hot(train_labels, depth=10)
test_labels = tf.one_hot(test_labels, depth=10)


# In[9]:


model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


# In[10]:


def custom_sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))


# In[11]:


def custom_accuracy(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    return accuracy


# In[12]:


model.compile(optimizer='adam',
              loss=custom_sparse_categorical_crossentropy,
              metrics=[custom_accuracy])


# In[13]:


model.fit(train_images, train_labels, epochs=5)


# In[14]:


test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy*100:.2f}%")


# In[15]:


plt.imshow(test_images[0])
prediction=model.predict(test_images)
print(np.argmax(prediction[0]))


# In[ ]:




