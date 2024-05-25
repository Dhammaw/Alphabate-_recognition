#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("datasets/A_Z Handwritten Data.csv")


# In[3]:


df


# In[4]:


x=df.drop("0",axis=1)
y=df["0"]


# In[5]:


y.nunique()    ,   y.shape


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


plt.imshow(x.iloc[17000].values.reshape(28,28))


# In[8]:


x=x/255


# In[9]:


from tensorflow.keras.utils import to_categorical


# In[10]:


ya=to_categorical(y,num_classes=26)


# In[11]:


ya.shape


# In[12]:


ya[:5]


# In[13]:


from tensorflow.keras.models import Sequential


# In[14]:


from tensorflow.keras.layers import Dense


# In[15]:


model=Sequential()


# In[16]:


model.add(Dense(16, input_shape=(784,), activation="relu"))


# In[17]:


model.add(Dense(16, activation="relu"))


# In[18]:


model.add(Dense(26, activation="softmax"))


# In[19]:


model.compile(loss="categorical_crossentropy", metrics=["accuracy"])


# In[20]:


model.summary()


# In[21]:


model.fit(x,ya, epochs=30, batch_size=256)


# In[22]:


temp=x.iloc[:5]
temp.shape


# In[23]:


model.predict(temp).argmax(axis=1)


# In[24]:


y[:5]


# In[25]:


y.nunique()


# # prediction on own alphabets

# In[26]:


import pandas as pd
import cv2


# In[27]:


A=cv2.imread("datasets/my alpha/z.jpg",0)


# In[28]:


A.shape


# In[29]:


A=cv2.resize(A,(28,28))


# In[30]:


A.shape


# In[31]:


A=A/255


# In[32]:


A=A.reshape(1,784)


# In[33]:


model.predict(A).argmax(axis=1)


# In[34]:


path='C:/Users/ADMIN/datasets/my alpha/'


# In[35]:


def pre_digi(filename):
    A=cv2.imread(path+filename,0)
    A=cv2.resize(A,(28,28))
    A=A/255
    A=A.reshape(1,784)
    return model.predict(A).argmax()


# In[36]:


pre_digi("b.jpg")


# In[37]:


from pathlib import Path
path: str = 'C:/Users/ADMIN/datasets/my alpha/'
osDir = Path(path)


# In[38]:


word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}


# In[39]:


import os
list_of_files=os.listdir(path)
for filename in list_of_files:
    yp=pre_digi(filename)
    print(filename,"t",word_dict[yp])


# In[40]:


i= 65
chr(i)


# In[ ]:




