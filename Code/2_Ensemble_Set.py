#!/usr/bin/env python
# coding: utf-8

#! n_jobs=-1 -> n_jobs=1

# In[ ]:

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import os
from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import random
from tqdm import trange
from scipy.stats import pointbiserialr
import math

warnings.filterwarnings('ignore')


# In[ ]:


# Load Data
train_raw = pd.read_csv('../Data/UNSW-NB15/train.csv')
print(train_raw.shape)
test_raw = pd.read_csv('../Data/UNSW-NB15/test.csv')
print(test_raw.shape)

# Seperate label and Drop ID
train_X = train_raw.drop(['id', 'attack_cat', 'label'], axis=1).select_dtypes(include='number')
train_Y = train_raw['label']
test_X = test_raw.drop(['id', 'attack_cat', 'label'], axis=1).select_dtypes(include='number')
test_Y = test_raw['label']

# Normalize data with min, max of training data
test_X1 = (test_X - train_X.min(axis=0)) / (train_X.max(axis=0) - train_X.min(axis=0))
train_X1 = (train_X - train_X.min(axis=0)) / (train_X.max(axis=0) - train_X.min(axis=0))

test_X1[test_X1 < 0] = 0
test_X1[test_X1 > 1] = 1


# In[ ]:


# correlation based feature selection
corr = train_X1.corr().abs()

threshold = 0.8
corr.values[np.tril_indices_from(corr.values)] = np.nan
redundant = []
for j in corr.columns:
    for i in corr.index:
        if corr.loc[i, j] > threshold:
            redundant.append((i, j))

train_X2 = train_X1.copy()
train_X2['label'] = train_Y
corr2 = train_X2.corr().abs()

corr3 = corr2['label'].iloc[:-1].copy()
drop = []

#! modify
for i, j in redundant:
    if corr3[i] > corr3[j]:
        if j not in drop:
            drop.append(j)
    elif i not in drop:
        drop.append(i)
print(drop)

#! 似乎沒有經過first stage處理
train_X1 = train_X1.drop(drop, axis=1)
test_X1 = test_X1.drop(drop, axis=1)
print(train_X1.shape)
print(test_X1.shape)


# In[ ]:


from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE, SequentialFeatureSelector, chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


# In[ ]:


subset_all = pd.read_csv('../Results/Feature_sets.csv').values


# In[ ]:


union_all = []
intersection_all = []
qourum_all = []

for k in trange(train_X1.shape[1]):
    unions = []
    intersections = []
    qourums = []
    for c in train_X1.columns:
        candidates = subset_all[:, :k+1]
        count = np.count_nonzero(candidates == c)
        if count > 0:
            unions.append(c)
        if count > len(subset_all) / 2:
            qourums.append(c)
        if count == len(subset_all):
            intersections.append(c)
    union_all.append(unions)
    intersection_all.append(intersections)
    qourum_all.append(qourums)
# print(union_all)
# print(intersection_all)
# print(qourum_all)
print([len(union_all[i]) for i in range(len(union_all))])
print([len(intersection_all[i]) for i in range(len(intersection_all))])
print([len(qourum_all[i]) for i in range(len(qourum_all))])


# In[ ]:


cv_times_all = []
f1_all = []
model = LogisticRegression(max_iter=10000, random_state=0, n_jobs=1)
for all in [union_all, intersection_all, qourum_all]:
    cv_times = []
    f1s = []
    for k in trange(train_X1.shape[1]):
        # cross validation
        if len(all[k]) > 0:
            second = time.time()
            #! modify, [:k+1] -> [k], use the kth list of 'all'.
            cv = cross_val_score(model, train_X1[all[k]], train_Y, scoring='f1', n_jobs=1)
            second2 = time.time()
            cv_times.append(second2 - second)
            f1s.append((cv.mean(), cv.std()))
        else:
            cv_times.append(0)
            f1s.append((0, 0))


    cv_times_all.append(cv_times)
    f1_all.append(f1s)


# In[ ]:


pd.DataFrame(cv_times_all, index=['union', 'intersection', 'quorum']).to_csv('../Results/Set_Time_LR.csv')
pd.DataFrame(f1_all, index=['union', 'intersection', 'quorum']).to_csv('../Results/Set_F1_LR.csv')


# In[ ]:


fig, axis = plt.subplots(1, 2, figsize=(12, 9))

plt.title('F1 Score and Time over number of features on Logistic Regression', loc='center')
plt.subplot(1, 2, 1)
plt.xlabel('Number of Features')
plt.ylabel('F1 Score')
plt.ylim((0, 1))

plt.plot(range(train_X1.shape[1]), np.array(f1_all)[0,:,0], color='blue', linestyle='-', label='union')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[1,:,0], color='red', linestyle='-', label='intersection')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[2,:,0], color='black', linestyle='-', label='quorum')

plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('Number of Features')
plt.ylabel('Time')

plt.plot(range(train_X1.shape[1]), cv_times_all[0], color='blue', linestyle='-', label='union')
plt.plot(range(train_X1.shape[1]), cv_times_all[1], color='red', linestyle='-', label='intersection')
plt.plot(range(train_X1.shape[1]), cv_times_all[2], color='black', linestyle='-', label='quorum')

plt.legend()

plt.tight_layout()
plt.savefig(os.path.join('2_Ensemble_Set', 'Set_LR.png'))


# In[ ]:


cv_times_all = []
f1_all = []
model = GradientBoostingClassifier(random_state=0)
for all in [union_all, intersection_all, qourum_all]:
    cv_times = []
    f1s = []
    for k in trange(train_X1.shape[1]):
        # cross validation
        if len(all[k]) > 0:
            second = time.time()
            #! modify, [:k+1] -> [k], use the kth list of 'all'.
            cv = cross_val_score(model, train_X1[all[k]], train_Y, scoring='f1', n_jobs=1)
            second2 = time.time()
            cv_times.append(second2 - second)
            f1s.append((cv.mean(), cv.std()))
        else:
            cv_times.append(0)
            f1s.append((0, 0))


    cv_times_all.append(cv_times)
    f1_all.append(f1s)


# In[ ]:


pd.DataFrame(cv_times_all, index=['union', 'intersection', 'quorum']).to_csv('../Results/Set_Time_GB.csv')
pd.DataFrame(f1_all, index=['union', 'intersection', 'quorum']).to_csv('../Results/Set_F1_GB.csv')


# In[ ]:


fig, axis = plt.subplots(1, 2, figsize=(12, 9))

plt.title('F1 Score and Time over number of features on Gradient Boosting', loc='center')
plt.subplot(1, 2, 1)
plt.xlabel('Number of Features')
plt.ylabel('F1 Score')
plt.ylim((0, 1))

plt.plot(range(train_X1.shape[1]), np.array(f1_all)[0,:,0], color='blue', linestyle='-', label='union')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[1,:,0], color='red', linestyle='-', label='intersection')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[2,:,0], color='black', linestyle='-', label='quorum')

plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('Number of Features')
plt.ylabel('Time')

plt.plot(range(train_X1.shape[1]), cv_times_all[0], color='blue', linestyle='-', label='union')
plt.plot(range(train_X1.shape[1]), cv_times_all[1], color='red', linestyle='-', label='intersection')
plt.plot(range(train_X1.shape[1]), cv_times_all[2], color='black', linestyle='-', label='quorum')

plt.legend()

plt.tight_layout()
plt.savefig(os.path.join('2_Ensemble_Set', 'Set_GB.png'))


# In[ ]:

os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras import Sequential, layers, losses, metrics, callbacks
from sklearn.model_selection import StratifiedKFold


# In[ ]:


def ModelCreate(input_shape):
    model = Sequential()
    model.add(layers.Dense(50, activation='relu', input_shape=input_shape))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
    return model


# In[ ]:


cv_times_all = []
f1_all = []
kf = StratifiedKFold(shuffle=True, random_state=0)
callback = callbacks.EarlyStopping(patience=3, min_delta=0.1, restore_best_weights=True)
for all in [union_all, intersection_all, qourum_all]:
    cv_times = []
    f1s = []
    for k in trange(train_X1.shape[1]):
        if len(all[k]) > 0:
            model = ModelCreate((len(all[k]),))
            # cross validation
            j = 0
            cv_time = 0
            cv = np.zeros(shape=5)
            #! modify, [:k+1] -> [k], use the kth list of 'all'.
            train_X2 = train_X1[all[k]].copy()  
            for train_index, test_index in kf.split(train_X2, train_Y):
                x_train_fold, x_test_fold = train_X2.iloc[train_index, :], train_X2.iloc[test_index, :]
                y_train_fold, y_test_fold = train_Y.iloc[train_index], train_Y.iloc[test_index]
                second = time.time()
                model.fit(x_train_fold.values, y_train_fold.values, validation_data=(x_test_fold, y_test_fold), epochs=30, callbacks=[callback], verbose=0)
                predict = model.predict(x_test_fold, use_multiprocessing=True)
                predict = np.where(predict < 0.5, 0, 1)
                cv[j] = f1_score(y_test_fold, predict)
                second2 = time.time()
                cv_time += second2 - second
                j += 1
            cv_times.append(cv_time)
            f1s.append((cv.mean(), cv.std()))
        else:
            cv_times.append(0)
            f1s.append((0, 0))
    
    cv_times_all.append(cv_times)
    f1_all.append(f1s)


# In[ ]:


pd.DataFrame(cv_times_all, index=['union', 'intersection', 'quorum']).to_csv('../Results/Set_Time_DNN.csv')
pd.DataFrame(f1_all, index=['union', 'intersection', 'quorum']).to_csv('../Results/Set_F1_DNN.csv')


# In[ ]:


fig, axis = plt.subplots(1, 2, figsize=(12, 9))

plt.title('F1 Score and Time over number of features on Deep Neuron Network', loc='center')
plt.subplot(1, 2, 1)
plt.xlabel('Number of Features')
plt.ylabel('F1 Score')
plt.ylim((0, 1))

plt.plot(range(train_X1.shape[1]), np.array(f1_all)[0,:,0], color='blue', linestyle='-', label='union')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[1,:,0], color='red', linestyle='-', label='intersection')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[2,:,0], color='black', linestyle='-', label='quorum')

plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('Number of Features')
plt.ylabel('Time')

plt.plot(range(train_X1.shape[1]), cv_times_all[0], color='blue', linestyle='-', label='union')
plt.plot(range(train_X1.shape[1]), cv_times_all[1], color='red', linestyle='-', label='intersection')
plt.plot(range(train_X1.shape[1]), cv_times_all[2], color='black', linestyle='-', label='quorum')

plt.legend()

plt.tight_layout()
plt.savefig(os.path.join('2_Ensemble_Set', 'Set_DNN.png'))

