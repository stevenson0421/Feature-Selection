import argparse
import numpy as np
from numpy import random
import pandas as pd

import matplotlib.pyplot as plt
import os

# GB, LR, DNN's score with whole feature
FULL_FEATURE_SCORE = {'GB':0.9358737434860567, 'LR':0.8303184157538102, 'DNN':0.9455613972040456}
SCORE_COLUMN = ['MD_score', 'MPR_score', 'MS_score']
FEATURE_SIZE = ['MaxDelta', 'MinPerfReq', 'MaxScore']
WIDTH = 0.25

def main():

    record_df = pd.read_csv('../Results/stopping_points.csv', index_col=0)
    GB = []
    LR = []
    DNN = []
    for index in record_df.index:
        if 'GB' in index:
            GB.append(index)
        elif 'LR' in index:
            LR.append(index)
        elif 'DNN' in index:
            DNN.append(index)
    tasks = {'GB':GB, 'LR':LR, 'DNN':DNN}

    # tackle score figures
    for key, value in tasks.items():
        score_df = record_df.loc[value][SCORE_COLUMN].copy()
        
        selection_methods = []
        for index in score_df.index:
            selection_methods.append(index.split('_')[2])

        plt.clf()
        plt.figure(figsize=(12, 9))
        plt.bar(range(len(score_df)), score_df['MD_score'], width=WIDTH, label=FEATURE_SIZE[0])
        plt.bar([x+WIDTH for x in range(len(score_df))], score_df['MPR_score'], width=WIDTH, label=FEATURE_SIZE[1])
        plt.bar([x+0.5 for x in range(len(score_df))], score_df['MS_score'], width=WIDTH, label=FEATURE_SIZE[2])

        plt.xticks([x+0.25 for x in range(len(score_df))], selection_methods)
        plt.xticks(rotation=20)
        plt.ylabel('F1 Score')
        plt.plot(FULL_FEATURE_SCORE[key])
        plt.plot(range(len(score_df)), [FULL_FEATURE_SCORE[key] for _ in range(len(score_df))], color='magenta', linestyle='--')
        print(score_df['MS_score'])
        

        plt.plot(range(len(score_df)), [score_df['MD_score']['Greedy_'+key+'_greedy'] for _ in range(len(score_df))], color='blue', linestyle='--')
        plt.plot(range(len(score_df)), [score_df['MPR_score']['Greedy_'+key+'_greedy'] for _ in range(len(score_df))], color='orange', linestyle='--')
        plt.plot(range(len(score_df)), [score_df['MS_score']['Greedy_'+key+'_greedy'] for _ in range(len(score_df))], color='green', linestyle='--')
        
        
        plt.legend(loc='lower left')
        filename = key + '_F1'
        plt.savefig('../Evaluation/' + filename + '.png')


    # tackle feature size figures
    for key, value in tasks.items():
        size_df = record_df.loc[value][FEATURE_SIZE].copy()
        
        selection_methods = []
        for index in size_df.index:
            selection_methods.append(index.split('_')[2])

        plt.clf()
        plt.figure(figsize=(12, 9))
        plt.bar(range(len(size_df)), size_df['MaxDelta'], width=WIDTH, label=FEATURE_SIZE[0])
        plt.bar([x+WIDTH for x in range(len(size_df))], size_df['MinPerfReq'], width=WIDTH, label=FEATURE_SIZE[1])
        plt.bar([x+0.5 for x in range(len(size_df))], size_df['MaxScore'], width=WIDTH, label=FEATURE_SIZE[2])
        plt.xticks([x+0.25 for x in range(len(size_df))], selection_methods)
        plt.xticks(rotation=20)
        plt.ylabel('Number of Features')
        plt.plot(FULL_FEATURE_SCORE[key])
        plt.legend(loc='lower left')

        plt.plot(range(len(size_df)), [size_df['MaxDelta']['Greedy_'+key+'_greedy'] for _ in range(len(size_df))], color='blue', linestyle='--')
        plt.plot(range(len(size_df)), [size_df['MinPerfReq']['Greedy_'+key+'_greedy'] for _ in range(len(size_df))], color='orange', linestyle='--')
        plt.plot(range(len(size_df)), [size_df['MaxScore']['Greedy_'+key+'_greedy'] for _ in range(len(size_df))], color='green', linestyle='--')


        filename = key + '_size'
        plt.savefig('../Evaluation/' + filename + '.png')



if __name__ == "__main__":
    main()