
# coding: utf-8

# In[2]:

import numpy as np
import scipy 
import matplotlib.pyplot as plt
import time


# In[3]:

import numpy as np
from sklearn.feature_selection import VarianceThreshold

# Loading Dataset
train_data = np.loadtxt('TinyMNIST/trainData.csv', dtype=np.float32, delimiter=',')
train_labels = np.loadtxt('TinyMNIST/trainLabels.csv', dtype=np.int32, delimiter=',')
test_data = np.loadtxt('TinyMNIST/testData.csv', dtype=np.float32, delimiter=',')
test_labels = np.loadtxt('TinyMNIST/testLabels.csv', dtype=np.int32, delimiter=',')
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Feature Selection
tr_samples_size, _ = train_data.shape
all_data = np.vstack((train_data,test_data))
sel = VarianceThreshold(threshold=0.90*(1-0.90))
all_data = sel.fit_transform(all_data)
train_data = all_data[:tr_samples_size]
test_data = all_data[tr_samples_size:]

tr_samples_size, feature_size = train_data.shape
te_samples_size, _ = test_data.shape
print('Train Data Samples:',tr_samples_size,
      ', Test Data Samples',te_samples_size,
      ', Feature Size(after feature-selection):', feature_size)


# In[4]:

types=[]
for i in range(10):
    types.append([])
for j in range(len(types)):
    for i in range(len(train_labels)):
        if train_labels[i]==j:
            types[j].append(train_data[i])

prior=[]
for i in types:
    prior.append(len(i)/5000)


# In[5]:

from sklearn import linear_model


# In[6]:

Class=[]
nclass=10
for i in range(nclass):
    Class.append([])
    for j in range(nclass):
        Class[i].append([])


# In[7]:

tstart=time.time()
for i in range(nclass):
    for j in range(i):
        clf = linear_model.SGDClassifier()
        X=types[i]+types[j]
        Y=[]
        for s in types[i]:
            Y.append(i)
        for s in types[j]:
            Y.append(j)
        clf.fit(X, Y)
        Class[i][j]=clf
print(time.time()-tstart)


# In[10]:

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


# In[12]:

tstart=time.time()
choose=[]
confusion=np.zeros([10,10])

for n  in range(len(test_data)):
    data=test_data[n]
    org=test_labels[n]
    versus=np.zeros(nclass)
    for i in range(nclass):
        for j in range(i):
            try:
                versus[Class[i][j].predict(data)]+=1
            except:
                1
    confusion[org][np.argmax(versus)]+=1
    choose.append(np.argmax(versus))
tend=time.time()-tstart


# In[13]:

c=[]
for i in range(len(test_data)):
    c.append(choose[i]==test_labels[i])
print(np.mean(c))


# In[15]:

tend


# In[ ]:



