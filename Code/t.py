import numpy as np
import pandas as pd

train_raw = pd.read_csv('../Data/UNSW-NB15/train.csv')
print(train_raw['dur'][0])
print(train_raw.loc[0, 'dur'])
print(train_raw.iloc[0, 1])

train_X = train_raw.drop(['id', 'attack_cat', 'label'], axis=1).select_dtypes(include='number')
train_Y = train_raw['label']

train_X1 = (train_X - train_X.min(axis=0)) / (train_X.max(axis=0) - train_X.min(axis=0))

corr = train_X1.corr()

# drop_train_raw[np.]


corr.values[np.tril_indices_from(corr.values)] = np.nan


redundants = []
for j in corr.columns:
    for i in corr.index:
        if corr[i][j] > 0.8:
            redundants.append((i, j))

# print(redundants)


train_X2 = train_X1.copy()
train_X2['label'] = train_Y
# corr2 = train_X2.corr().abs()
corr2 = train_X2.corr().abs()
# corr3 = corr2['label'].iloc[:-1].copy()
corr3 = corr2['label'].iloc[:-1].copy()
drop = []
for i, j in redundants:
    if i == 'ackdat' or j == 'ackdat':
        print(i, j)
    if corr3[i] > corr3[j]:
        drop.append(j)
        if j == 'ackdat':
            print(i, j)
    else:
        drop.append(i)
        if i == 'ackdat':
            print(j, i)

print(set(drop))