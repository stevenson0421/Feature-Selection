#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


files = ['Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
'Monday-WorkingHours.pcap_ISCX.csv',
'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
'Tuesday-WorkingHours.pcap_ISCX.csv',
'Wednesday-workingHours.pcap_ISCX.csv']

d = pd.DataFrame()
for f in files:
    raw = pd.read_csv('../Data/CICIDS-2017/' + f)
    d = pd.concat([d, raw], axis=0)

train = pd.DataFrame()
for i in range(5):
    train = pd.concat([train, d.sample(100000)], axis=0)
train.to_csv('../Data/CICIDS-2017/train.csv')
print(train.shape)
test= pd.DataFrame()
for i in range(5):
    test = pd.concat([test, d.sample(100000)], axis=0)
test.to_csv('../Data/CICIDS-2017/test.csv')
print(test.shape)

