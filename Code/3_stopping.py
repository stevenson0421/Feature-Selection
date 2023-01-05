import argparse
import numpy as np
from numpy import random
import pandas as pd

import matplotlib.pyplot as plt
import os

tasks = ['Individual', 'Set', 'Greedy']
models = ['LR', 'GB', 'DNN']



def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--method", default='MD', help="e.g. MD(MaxDelta), MPR(MinPerfReq), MS(MaxScore)", type=str)
    parser.add_argument("-t", "--tolerence", default=0.03, help="Tolerence rate for MPR.", type=float)
    parser.add_argument("-r", "--rho", default=0.003, help="Penalty coefficient for MS.", type=float)
    args = parser.parse_args()

    stop_df_all = []
    index_all = []
    for i in range(len(tasks)):
        for j in range(len(models)):

            filename = tasks[i] + '_F1_' + models[j]
            # read the results, extract their score parts.
            record_df = pd.read_csv('./Results/KDDCUP99/' + filename + '.csv', index_col=0)
            record2_df = pd.read_csv('./Results/KDDCUP99/' + filename + '_Test.csv', index_col=0)
            score_df = pd.DataFrame(index=record_df.index)
            test_df = pd.DataFrame(index=record_df.index)
            time_df = pd.DataFrame(index=record_df.index)
            size_df = pd.DataFrame(index=record_df.index)
            for column in record_df.columns:
                for index in record_df.index:
                    text_tuple = record_df.loc[index][column]
                    test = record2_df.loc[index][column]
                    if tasks[i] == 'Set':
                        score, _, size = text2tuple(text_tuple, tasks[i])
                        size_df.at[index, column] = size
                    else:
                        score, _ = text2tuple(text_tuple, tasks[i])
                    score_df.at[index, column] = score
                    test_df.at[index, column] = test
            size_df = size_df.astype(int)
            # mv F1_png method_1/F1_png
            # mv MultiBar method_1/MultiBar
            # mv Time_png method_1/Time_png


            plt.clf()
            for index in record_df.index:
                index_score = score_df.loc[index].values
                index_test = test_df.loc[index].values
                # Get stopping points by the following methods.
                stop_index1 = MaxDelta(index_score, args, tasks[i], size_df.loc[index])
                stop_index2 = MinPerfReq(index_score, args, tasks[i], size_df.loc[index])
                stop_index3 = MaxScore(index_score, args, tasks[i], size_df.loc[index])

                # save the stopping feature size by different methods
                stop_acc1 = index_score[stop_index1]
                stop_acc2 = index_score[stop_index2]
                stop_acc3 = index_score[stop_index3]
                stop_test1 = index_test[stop_index1]
                stop_test2 = index_test[stop_index2]
                stop_test3 = index_test[stop_index3]
                index_all.append(tasks[i] + '_' + models[j] + '_' + index)
                if tasks[i] == 'Set':
                    info = [size_df.loc[index][str(stop_index1)], stop_acc1, stop_test1, size_df.loc[index][str(stop_index2)], stop_acc2, stop_test2, size_df.loc[index][str(stop_index3)], stop_acc3, stop_test3]
                else:
                    info = [stop_index1+1, stop_acc1, stop_test1, stop_index2+1, stop_acc2, stop_test2, stop_index3+1, stop_acc3, stop_test3]
                stop_df_all.append(info)


    print(len(stop_df_all))
    pd.DataFrame(stop_df_all, index=index_all, columns=['MaxDelta', 'MD_score', 'MD_test', 'MinPerfReq', 'MPR_score', 'MPR_test', 'MaxScore', 'MS_score', 'MS_test']).to_csv('./Results/KDDCUP99/stopping_points.csv')


def text2tuple(text, task):
    info = text[1:-1].split(', ')
    if task == 'Set':
        return float(info[0]), float(info[1]), int(info[2])
    else:
        return float(info[0]), float(info[1])


def MaxDelta(score, args, task, size_df):
    max_delta = 0
    index = len(score) - 1
    for i in range(len(score)-1, 0, -1):
        delta = score[i] - score[i-1]
        if delta > max_delta and score[i-1] != 0:
            max_delta = delta
            index = i

    return index

def MinPerfReq(score, args, task, size_df):
    best_CVscore = score[-1]
    index = len(score) - 1
    for i in range(len(score)-1, 0, -1):
        delta = (best_CVscore - score[i]) / best_CVscore
        if delta > args.tolerence:
            index = i
            break

    return index

def MaxScore(score, args, task, size_df):
    best_performance = 0
    index = len(score) - 1
    for i in range(len(score)-1, 0, -1):
        current_size = i + 1
        adj_score = score[i] - (args.rho * current_size)
        if adj_score > best_performance and score[i-1] != 0:
            best_performance = adj_score
            index = i

    return index


if __name__ == "__main__":
    main()