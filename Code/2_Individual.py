#!/usr/bin/env python
# coding: utf-8

#! n_jobs=-1 -> n_jobs=1

# In[1]:

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import warnings
import time
import os
import random
from tqdm import trange
from scipy.stats import pointbiserialr
import ctypes
import math

warnings.filterwarnings('ignore')


# In[2]:


# Load Data
train_raw = pd.read_csv('../Data/UNSW-NB15/train.csv')
print(train_raw.shape)
test_raw = pd.read_csv('../Data/UNSW-NB15/test.csv')
print(test_raw.shape)


# In[3]:


# Seperate label and Drop ID
train_X = train_raw.drop(['id', 'attack_cat', 'label'], axis=1).select_dtypes(include='number')
train_Y = train_raw['label']
test_X = test_raw.drop(['id', 'attack_cat', 'label'], axis=1).select_dtypes(include='number')
test_Y = test_raw['label']


# In[4]:


# Normalize data with min, max of training data
test_X1 = (test_X - train_X.min(axis=0)) / (train_X.max(axis=0) - train_X.min(axis=0))
train_X1 = (train_X - train_X.min(axis=0)) / (train_X.max(axis=0) - train_X.min(axis=0))

test_X1[test_X1 < 0] = 0
test_X1[test_X1 > 1] = 1


# In[5]:


# correlation heatmap
corr = train_X1.corr().abs()

plt.figure(figsize=(12, 9))
sns.heatmap(corr)
plt.tight_layout()
plt.savefig(os.path.join('2_Individual', 'correlation_heatmap.png'))


# In[6]:


# correlation rankings between features and label
train_X2 = train_X1.copy()
train_X2['label'] = train_Y

corr2 = train_X2.corr().abs()

plt.figure(figsize=(12, 9))
sns.barplot(x=corr2['label'].iloc[:-1], y=corr2.columns[:-1], order=corr2['label'].iloc[:-1].sort_values(ascending=False).index)
plt.tight_layout()
plt.savefig(os.path.join('2_Individual', 'correlation_bar.png'))


# In[7]:


# consider redundant if correlation > threshold
threshold = 0.8
corr.values[np.tril_indices_from(corr.values)] = np.nan
redundant = []
for j in corr.columns:
    for i in corr.index:
        if corr.loc[i, j] > threshold:
            redundant.append((i, j))
print(redundant)


# In[8]:


# select redundant columns to drop
corr3 = corr2['label'].iloc[:-1].copy()
drop = []

for i, j in redundant:
    #! modify
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


# In[9]:


from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE, SequentialFeatureSelector, chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


# In[10]:


# select 1 best feature iteratively with chi2, ANOVA, mutual info
subset_all = []
for sf in [chi2, f_classif, mutual_info_classif]:
    cols = []
    selector = SelectKBest(sf, k=1)
    for k in trange(train_X1.shape[1]):
        # select one best feature and add it to subset
        selector.fit(train_X1.drop(cols, axis=1), train_Y)
        scores = selector.pvalues_
        #! PROBLEM, check if right.
        index = np.argsort(scores)[::-1][-1]
        f = train_X1.columns.drop(cols)[index]
        # f = selector.get_feature_names_out(train_X1.columns.drop(cols))
        cols.append(f)
    subset_all.append(cols)
    print(cols)


# In[11]:


# select 1 best feature iteratively with SFS, using RF, LR
for model in [RandomForestClassifier(random_state=0, n_jobs=1), LogisticRegression(max_iter=10000, random_state=0, n_jobs=1)]:
    cols = []
    selector = SequentialFeatureSelector(model, n_features_to_select=1, scoring='f1', cv=5, n_jobs=1)
    for k in trange(train_X1.shape[1]-1):
        selector.fit(train_X1.drop(cols, axis=1), train_Y)
        f = train_X1.columns.drop(cols)[selector.get_support()]
        cols.append(f[0])
    cols.append(train_X1.columns.drop(cols)[0])
    subset_all.append(cols)
    print(cols)
print(len(subset_all))


# In[12]:


# select 1 best feature iteratively with importance, using RF, LR
for model in [RandomForestClassifier(random_state=0, n_jobs=1), LogisticRegression(max_iter=10000, random_state=0, n_jobs=1)]:
    cols = []
    selector = SelectFromModel(model, threshold=-np.inf, max_features=1)
    for k in trange(train_X1.shape[1]-1):
        selector.fit(train_X1.drop(cols, axis=1), train_Y)
        f = train_X1.columns.drop(cols)[selector.get_support()]
        cols.append(f[0])
    cols.append(train_X1.columns.drop(cols)[0])
    subset_all.append(cols)
    print(cols)


# In[13]:


# save selected feature sets to csv file
pd.DataFrame(subset_all, index=['chi2', 'ANOVA', 'mutualinfo', 'sfs(rf)', 'sfs(lr)', 'im(rf)', 'im(lr)']).to_csv('../Results/Feature_sets.csv')


# In[14]:


# measure performance by cv(f1 score)
cv_times_all = []
f1_all = []
model = LogisticRegression(max_iter=10000, random_state=0, n_jobs=1)
for i in range(len(subset_all)):
    cv_times = []
    f1s = []
    for k in trange(train_X1.shape[1]):
        # cross validation
        second = time.time()
        cv = cross_val_score(model, train_X1[subset_all[i][:k+1]], train_Y, scoring='f1', n_jobs=1)
        second2 = time.time()
        cv_times.append(second2 - second)
        f1s.append((cv.mean(), cv.std()))
    
    cv_times_all.append(cv_times)
    f1_all.append(f1s)


# In[15]:


pd.DataFrame(cv_times_all, index=['chi2', 'ANOVA', 'mutualinfo', 'sfs(rf)', 'sfs(lr)', 'im(rf)', 'im(lr)']).to_csv('../Results/Time_LR.csv')
pd.DataFrame(f1_all, index=['chi2', 'ANOVA', 'mutualinfo', 'sfs(rf)', 'sfs(lr)', 'im(rf)', 'im(lr)']).to_csv('../Results/F1_LR.csv')


# In[16]:


fig, axis = plt.subplots(1, 2, figsize=(12, 9))

plt.title('F1 Score and Time over number of features on Logistic Regression', loc='center')
plt.subplot(1, 2, 1)
plt.xlabel('Number of Features')
plt.ylabel('F1 Score')
plt.ylim((0, 1))

plt.plot(range(train_X1.shape[1]), np.array(f1_all)[0,:,0], color='blue', linestyle='-', label='chi2')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[1,:,0], color='red', linestyle='-', label='ANOVA')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[2,:,0], color='black', linestyle='-', label='mutual information')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[3,:,0], color='cyan', linestyle='-', label='sfs(rf)')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[4,:,0], color='darkcyan', linestyle='-', label='sfs(lr)')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[5,:,0], color='green', linestyle='-', label='im(rf)')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[6,:,0], color='darkgreen', linestyle='-', label='im(lr)')

plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('Number of Features')
plt.ylabel('Time')

plt.plot(range(train_X1.shape[1]), cv_times_all[0], color='blue', linestyle='-', label='chi2')
plt.plot(range(train_X1.shape[1]), cv_times_all[1], color='red', linestyle='-', label='ANOVA')
plt.plot(range(train_X1.shape[1]), cv_times_all[2], color='black', linestyle='-', label='mutual information')
plt.plot(range(train_X1.shape[1]), cv_times_all[3], color='cyan', linestyle='-', label='sfs(rf)')
plt.plot(range(train_X1.shape[1]), cv_times_all[4], color='darkcyan', linestyle='-', label='sfs(lr)')
plt.plot(range(train_X1.shape[1]), cv_times_all[5], color='green', linestyle='-', label='im(rf)')
plt.plot(range(train_X1.shape[1]), cv_times_all[6], color='darkgreen', linestyle='-', label='im(lr)')

plt.legend()

plt.tight_layout()
plt.savefig(os.path.join('2_Individual', 'Individual_LR.png'))


# In[17]:


cv_times_all = []
f1_all = []

model = GradientBoostingClassifier(random_state=0)
for i in range(len(subset_all)):
    cv_times = []
    f1s = []
    for k in trange(train_X1.shape[1]):
        # cross validation
        second = time.time()
        cv = cross_val_score(model, train_X1[subset_all[i][:k+1]], train_Y, scoring='f1', n_jobs=1)
        second2 = time.time()
        cv_times.append(second2 - second)
        f1s.append((cv.mean(), cv.std()))
    
    cv_times_all.append(cv_times)
    f1_all.append(f1s)


# In[18]:


pd.DataFrame(cv_times_all, index=['chi2', 'ANOVA', 'mutualinfo', 'sfs(rf)', 'sfs(lr)', 'im(rf)', 'im(lr)']).to_csv('../Results/Time_GB.csv')
pd.DataFrame(f1_all, index=['chi2', 'ANOVA', 'mutualinfo', 'sfs(rf)', 'sfs(lr)', 'im(rf)', 'im(lr)']).to_csv('../Results/F1_GB.csv')


# In[19]:


fig, axis = plt.subplots(1, 2, figsize=(12, 9))

plt.title('F1 Score and Time over number of features on Gradient Boosting', loc='center')
plt.subplot(1, 2, 1)
plt.xlabel('Number of Features')
plt.ylabel('F1 Score')
plt.ylim((0, 1))

plt.plot(range(train_X1.shape[1]), np.array(f1_all)[0,:,0], color='blue', linestyle='-', label='chi2')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[1,:,0], color='red', linestyle='-', label='ANOVA')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[2,:,0], color='black', linestyle='-', label='mutual information')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[3,:,0], color='cyan', linestyle='-', label='sfs(rf)')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[4,:,0], color='darkcyan', linestyle='-', label='sfs(lr)')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[5,:,0], color='green', linestyle='-', label='im(rf)')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[6,:,0], color='darkgreen', linestyle='-', label='im(lr)')

plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('Number of Features')
plt.ylabel('Time')

plt.plot(range(train_X1.shape[1]), cv_times_all[0], color='blue', linestyle='-', label='chi2')
plt.plot(range(train_X1.shape[1]), cv_times_all[1], color='red', linestyle='-', label='ANOVA')
plt.plot(range(train_X1.shape[1]), cv_times_all[2], color='black', linestyle='-', label='mutual information')
plt.plot(range(train_X1.shape[1]), cv_times_all[3], color='cyan', linestyle='-', label='sfs(rf)')
plt.plot(range(train_X1.shape[1]), cv_times_all[4], color='darkcyan', linestyle='-', label='sfs(lr)')
plt.plot(range(train_X1.shape[1]), cv_times_all[5], color='green', linestyle='-', label='im(rf)')
plt.plot(range(train_X1.shape[1]), cv_times_all[6], color='darkgreen', linestyle='-', label='im(lr)')

plt.legend()

plt.tight_layout()
plt.savefig(os.path.join('2_Individual', 'Individual_GB.png'))


# In[20]:


from keras import Sequential, layers, optimizers, losses, metrics, callbacks
from sklearn.model_selection import StratifiedKFold


# In[21]:


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


# In[22]:
subset_all = pd.read_csv('../Results/Feature_sets.csv', index_col=0).values

cv_times_all = []
f1_all = []

kf = StratifiedKFold(shuffle=True, random_state=0)
callback = callbacks.EarlyStopping(patience=3, min_delta=0.1, restore_best_weights=True)
for i in range(len(subset_all)):
    cv_times = []
    f1s = []
    for k in trange(train_X1.shape[1]):
        model = ModelCreate((k+1,))
        # cross validation
        j = 0
        cv_time = 0
        cv = np.zeros(shape=5)
        train_X2 = train_X1[subset_all[i][:k+1]].copy()
        print(train_X2.shape)
        for train_index, test_index in kf.split(train_X2, train_Y):
            x_train_fold, x_test_fold = train_X2.iloc[train_index, :], train_X2.iloc[test_index, :]
            y_train_fold, y_test_fold = train_Y.iloc[train_index], train_Y.iloc[test_index]

            second = time.time()
            model.fit(x_train_fold.values, y_train_fold.values, validation_data=(x_test_fold, y_test_fold), epochs=30, callbacks=[callback], use_multiprocessing=True, verbose=0)
            predict = model.predict(x_test_fold, use_multiprocessing=True)
            predict = np.where(predict < 0.5, 0, 1)
            cv[j] = f1_score(y_test_fold, predict)
            second2 = time.time()
            cv_time += second2 - second
            j += 1
        cv_times.append(cv_time)
        f1s.append((cv.mean(), cv.std()))
    
    cv_times_all.append(cv_times)
    f1_all.append(f1s)


# In[23]:


pd.DataFrame(cv_times_all, index=['chi2', 'ANOVA', 'mutualinfo', 'sfs(rf)', 'sfs(lr)', 'im(rf)', 'im(lr)']).to_csv('../Results/Time_DNN.csv')
pd.DataFrame(f1_all, index=['chi2', 'ANOVA', 'mutualinfo', 'sfs(rf)', 'sfs(lr)', 'im(rf)', 'im(lr)']).to_csv('../Results/F1_DNN.csv')


# In[24]:


fig, axis = plt.subplots(1, 2, figsize=(12, 9))

plt.title('F1 Score and Time over number of features on Deep Neural Network', loc='center')
plt.subplot(1, 2, 1)
plt.xlabel('Number of Features')
plt.ylabel('F1 Score')
plt.ylim((0, 1))

plt.plot(range(train_X1.shape[1]), np.array(f1_all)[0,:,0], color='blue', linestyle='-', label='chi2')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[1,:,0], color='red', linestyle='-', label='ANOVA')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[2,:,0], color='black', linestyle='-', label='mutual information')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[3,:,0], color='cyan', linestyle='-', label='sfs(rf)')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[4,:,0], color='darkcyan', linestyle='-', label='sfs(lr)')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[5,:,0], color='green', linestyle='-', label='im(rf)')
plt.plot(range(train_X1.shape[1]), np.array(f1_all)[6,:,0], color='darkgreen', linestyle='-', label='im(lr)')

plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('Number of Features')
plt.ylabel('Time')

plt.plot(range(train_X1.shape[1]), cv_times_all[0], color='blue', linestyle='-', label='chi2')
plt.plot(range(train_X1.shape[1]), cv_times_all[1], color='red', linestyle='-', label='ANOVA')
plt.plot(range(train_X1.shape[1]), cv_times_all[2], color='black', linestyle='-', label='mutual information')
plt.plot(range(train_X1.shape[1]), cv_times_all[3], color='cyan', linestyle='-', label='sfs(rf)')
plt.plot(range(train_X1.shape[1]), cv_times_all[4], color='darkcyan', linestyle='-', label='sfs(lr)')
plt.plot(range(train_X1.shape[1]), cv_times_all[5], color='green', linestyle='-', label='im(rf)')
plt.plot(range(train_X1.shape[1]), cv_times_all[6], color='darkgreen', linestyle='-', label='im(lr)')

plt.legend()

plt.tight_layout()
plt.savefig(os.path.join('2_Individual', 'Individual_DNN.png'))
