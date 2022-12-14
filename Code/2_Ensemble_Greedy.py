#!/usr/bin/env python
# coding: utf-8

#! n_jobs=1 -> Model

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
from sklearn.model_selection import cross_val_score, StratifiedKFold


# # In[ ]:


# # greedy search
# greedy_all = []
# score_functions = [chi2, f_classif, mutual_info_classif]
# clfs = [RandomForestClassifier(random_state=0, n_jobs=1), LogisticRegression(max_iter=10000, random_state=0, n_jobs=1)]
# Model = LogisticRegression(max_iter=10000, random_state=0, n_jobs=1)
# #! Original greedy_all's maximum length is up to 22, but 24 is correct.
# #! modify, train_X1.shape[1]-1 -> train_X1.shape[1]
# for k in trange(train_X1.shape[1]-1):
#     features = []
#     scores = []
#     for sf in score_functions:
#         selector = SelectKBest(sf, k=1)
#         # select one best feature and add it to subset
#         selector.fit(train_X1.drop(greedy_all, axis=1), train_Y)
#         #! modify
#         score = selector.pvalues_
#         index = np.argsort(score)[::-1][-1]
#         f = train_X1.columns.drop(greedy_all)[index]
#         # f = selector.get_feature_names_out(train_X1.columns.drop(greedy_all))
#         features.append(f)
#         cv = cross_val_score(Model, train_X1[greedy_all+[f]], train_Y, scoring='f1', n_jobs=1)
#         scores.append(cv.mean())
    
#     for clf in clfs:
#         selector = SequentialFeatureSelector(clf, n_features_to_select=1, scoring='f1', cv=5, n_jobs=1)
#         # select one best feature and add it to subset
#         selector.fit(train_X1.drop(greedy_all, axis=1), train_Y)
#         f = train_X1.columns.drop(greedy_all)[selector.get_support()]
#         features.append(f[0])
#         cv = cross_val_score(Model, train_X1[greedy_all+[f[0]]], train_Y, scoring='f1', n_jobs=1)
#         scores.append(cv.mean())

#     for clf in clfs:
#         selector = SelectFromModel(clf, threshold=-np.inf, max_features=1)
#         # select one best feature and add it to subset
#         selector.fit(train_X1.drop(greedy_all, axis=1), train_Y)
#         f = train_X1.columns.drop(greedy_all)[selector.get_support()]
#         features.append(f[0])
#         cv = cross_val_score(Model, train_X1[greedy_all+[f[0]]], train_Y, scoring='f1', n_jobs=1)
#         scores.append(cv.mean())

#     i_best = np.argmax(scores)
#     greedy_all.append(features[i_best])
#     print(len(greedy_all))


# pd.DataFrame([greedy_all], index=['greedy']).to_csv('../Results/Greedy_Feature_sets.csv')


# # In[ ]:


# # test with LR
# cv_times_all = []
# f1_all = []
# Model = LogisticRegression(max_iter=10000, random_state=0, n_jobs=1)
# for k in trange(train_X1.shape[1]):
#     # cross validation
#     second = time.time()
#     cv = cross_val_score(Model, train_X1[greedy_all[:k+1]], train_Y, scoring='f1', n_jobs=1)
#     second2 = time.time()
#     cv_times_all.append(second2 - second)
#     f1_all.append((cv.mean(), cv.std()))


# # In[ ]:


# pd.DataFrame([cv_times_all], index=['greedy']).to_csv('../Results/Greedy_Time_LR.csv')
# pd.DataFrame([f1_all], index=['greedy']).to_csv('../Results/Greedy_F1_LR.csv')


# # In[ ]:


# fig, axis = plt.subplots(1, 2, figsize=(12, 9))

# plt.title('F1 Score and Time over number of features on Logistic Regression', loc='center')
# plt.subplot(1, 2, 1)
# plt.xlabel('Number of Features')
# plt.ylabel('F1 Score')
# plt.ylim((0, 1))

# plt.plot(range(train_X1.shape[1]), np.array(f1_all)[:,0], color='blue', linestyle='-', label='greedy')

# plt.legend()

# plt.subplot(1, 2, 2)
# plt.xlabel('Number of Features')
# plt.ylabel('Time')

# plt.plot(range(train_X1.shape[1]), cv_times_all, color='blue', linestyle='-', label='greedy')

# plt.legend()

# plt.tight_layout()
# plt.savefig(os.path.join('2_Ensemble_Greedy', 'Greedy_LR.png'))


# # In[ ]:


# # test with GB
# cv_times_all = []
# f1_all = []
# Model = GradientBoostingClassifier(random_state=0)
# for k in trange(train_X1.shape[1]):
#     # cross validation
#     second = time.time()
#     cv = cross_val_score(Model, train_X1[greedy_all[:k+1]], train_Y, scoring='f1', n_jobs=1)
#     second2 = time.time()
#     cv_times_all.append(second2 - second)
#     f1_all.append((cv.mean(), cv.std()))


# # In[ ]:


# pd.DataFrame([cv_times_all], index=['greedy']).to_csv('../Results/Greedy_Time_GB.csv')
# pd.DataFrame([f1_all], index=['greedy']).to_csv('../Results/Greedy_F1_GB.csv')


# # In[ ]:


# fig, axis = plt.subplots(1, 2, figsize=(12, 9))

# plt.title('F1 Score and Time over number of features on Gradient Boosting', loc='center')
# plt.subplot(1, 2, 1)
# plt.xlabel('Number of Features')
# plt.ylabel('F1 Score')
# plt.ylim((0, 1))

# plt.plot(range(train_X1.shape[1]), np.array(f1_all)[:,0], color='blue', linestyle='-', label='greedy')

# plt.legend()

# plt.subplot(1, 2, 2)
# plt.xlabel('Number of Features')
# plt.ylabel('Time')

# plt.plot(range(train_X1.shape[1]), cv_times_all, color='blue', linestyle='-', label='greedy')

# plt.legend()

# plt.tight_layout()
# plt.savefig(os.path.join('2_Ensemble_Greedy', 'Greedy_GB.png'))


# In[ ]:

# from keras import Sequential, layers, losses, metrics, callbacks


# # In[ ]:


# def ModelCreate(input_shape):
#     Model = Sequential()
#     Model.add(layers.Dense(50, activation='relu', input_shape=input_shape))
#     Model.add(layers.Dropout(0.2))
#     Model.add(layers.Dense(50, activation='relu'))
#     Model.add(layers.Dropout(0.2))
#     Model.add(layers.Dense(50, activation='relu'))
#     Model.add(layers.Dropout(0.2))
#     Model.add(layers.Dense(50, activation='relu'))
#     Model.add(layers.Dropout(0.2))
#     Model.add(layers.Dense(1, activation='sigmoid'))
#     Model.compile(optimizer='adam', loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
#     return Model


# # In[ ]:

# greedy_all = pd.read_csv('../Results/Greedy_Feature_sets.csv', index_col=0).values[0]

# cv_times_all = []
# f1_all = []
# kf = StratifiedKFold(shuffle=True, random_state=0)
# callback = callbacks.EarlyStopping(patience=3, min_delta=0.1, restore_best_weights=True)
# for k in trange(train_X1.shape[1]-1):
#     Model = ModelCreate((k+1,))
#     # cross validation
#     j = 0
#     cv_time = 0
#     cv = np.zeros(shape=5)
#     train_X2 = train_X1[greedy_all[:k+1]].copy()
#     print(train_X2.shape, k+1)
#     for train_index, test_index in kf.split(train_X2, train_Y):
#         x_train_fold, x_test_fold = train_X2.iloc[train_index, :], train_X2.iloc[test_index, :]
#         y_train_fold, y_test_fold = train_Y.iloc[train_index], train_Y.iloc[test_index]

#         second = time.time()
#         Model.fit(x_train_fold.values, y_train_fold.values, validation_data=(x_test_fold, y_test_fold), epochs=30, callbacks=[callback], verbose=0)
#         predict = Model.predict(x_test_fold, use_multiprocessing=True)
#         predict = np.where(predict < 0.5, 0, 1)
#         cv[j] = f1_score(y_test_fold, predict)
#         second2 = time.time()
#         cv_time += second2 - second
#         j += 1
#     cv_times_all.append(cv_time)
#     f1_all.append((cv.mean(), cv.std()))


# # In[ ]:


# pd.DataFrame([cv_times_all], index=['greedy']).to_csv('../Results/Greedy_Time_DNN.csv')
# pd.DataFrame([f1_all], index=['greedy']).to_csv('../Results/Greedy_F1_DNN.csv')


# In[ ]:


cv_times_all = pd.read_csv('../Results/Greedy_Time_DNN.csv', index_col=0).values[0]
f1_all = pd.read_csv('../Results/Greedy_F1_DNN.csv', index_col=0).values[0]
f1_all = [f1_all[i] for i in range(len(f1_all))]
print(f1_all[0][0])


fig, axis = plt.subplots(1, 2, figsize=(12, 9))

plt.title('F1 Score and Time over number of features on Deep Neuron Network', loc='center')
plt.subplot(1, 2, 1)
plt.xlabel('Number of Features')
plt.ylabel('F1 Score')
plt.ylim((0, 1))

plt.plot(range(train_X1.shape[1]-1), np.array(f1_all)[:,0], color='blue', linestyle='-', label='greedy')

plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('Number of Features')
plt.ylabel('Time')

plt.plot(range(train_X1.shape[1]-1), cv_times_all, color='blue', linestyle='-', label='greedy')

plt.legend()

plt.tight_layout()
plt.savefig(os.path.join('2_Ensemble_Greedy', 'Greedy_DNN.png'))

