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

tasks = ['LR', 'GB', 'DNN']
methods = ['union', 'intersection', 'quorum']

def main():
    set_df = pd.read_csv('./Results/KDDCUP99/Set_Feature_sets.csv', index_col=0)
    for task in tasks:
        method_df = pd.read_csv('./Results/KDDCUP99/_Set_F1_'+task+'.csv', index_col=0)
        for method in methods:
            feature = set_df.loc[method].values

            # print([len(feature[i]) for i in range(len(feature))])

            for i in range(len(feature)):
                cur_feature_list = text2list(feature[i])
                text_tuple = method_df.loc[method, str(i)]
                score, dev = text2tuple(text_tuple)
                print(cur_feature_list)
                print(len(cur_feature_list))
                method_df.at[method, str(i)] = (score, dev, len(cur_feature_list))
                print(method_df.loc[method, str(i)])

            for i in range(1, len(feature)-1):
                last_feature_list = text2list(feature[i-1])
                cur_feature_list = text2list(feature[i])
                next_feature_list = text2list(feature[i+1])
                print(len(cur_feature_list), end=' ')

                if len(cur_feature_list) != len(last_feature_list) and len(cur_feature_list) == len(next_feature_list):
                    method_df.at[method, str(i)] = method_df.loc[method, str(i-1)]

                elif len(cur_feature_list) == len(last_feature_list) and len(cur_feature_list) == len(next_feature_list):
                    # print(method_df)
                    method_df.at[method, str(i)] = method_df.loc[method, str(i-1)]

        method_df.to_csv('./Results/KDDCUP99/Set_F1_'+task+'.csv')


def text2list(text):
    info = text[1:-1].split(', ')
    for i in range(len(info)):
        info[i] = info[i][1:-1]
    if info[0] == '':
        return []
    else:
        return info

def text2tuple(text):
    info = text[1:-1].split(', ')
    return float(info[0]), float(info[1])

if __name__ == '__main__':
    main()
