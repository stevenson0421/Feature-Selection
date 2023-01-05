import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

tasks = ['Individual', 'Set', 'Greedy']
models = ['LR', 'GB', 'DNN']

def main():
    stop_df_all = []
    index_all = []
    for i in range(len(tasks)):
        for j in range(len(models)):

            F1_filename = tasks[i] + '_F1_' + models[j]
            # read the results, extract their score parts.
            F1_df = pd.read_csv('./Results/KDDCUP99/' + F1_filename + '.csv', index_col=0)

            score_df = pd.DataFrame(index=F1_df.index)
            size_df = pd.DataFrame(index=F1_df.index)

            for column in F1_df.columns:
                for index in F1_df.index:
                    text_tuple = F1_df.loc[index][column]
                    if tasks[i] == 'Set':
                        score, _, size = text2tuple(text_tuple, tasks[i])
                        size_df.at[index, column] = size
                    else:
                        score, _ = text2tuple(text_tuple, tasks[i])
                    score_df.at[index, column] = score

            size_df = size_df.astype(int)

            plt.clf()
            for index in F1_df.index:
                index_score = score_df.loc[index].values
                # save the stopping feature size by different methods
                index_all.append(tasks[i] + '_' + models[j] + '_' + index)
                plt.plot(range(1, len(index_score)+1), index_score,  linestyle='-', label=index)

            plt.xlabel('Number of Features')
            plt.ylabel('F1 Score')
            # plt.xlim((1, 25))
            # plt.ylim((0.6, 1))
            plt.legend()
            plt.grid()
            if not os.path.exists('./Evaluation/KDDCUP99/F1_png/'):
                os.makedirs('./Evaluation/KDDCUP99/F1_png/')
            plt.savefig(os.path.join('./Evaluation/KDDCUP99/F1_png', F1_filename + '.png'))

            F1_filename = tasks[i] + '_F1_' + models[j]
            # read the results, extract their score parts.
            F1_df = pd.read_csv('./Results/KDDCUP99/' + F1_filename + '_Test.csv', index_col=0)

            score_df = pd.DataFrame(index=F1_df.index)
            size_df = pd.DataFrame(index=F1_df.index)

            for column in F1_df.columns:
                for index in F1_df.index:
                    score = F1_df.loc[index][column]
                    score_df.at[index, column] = score

            size_df = size_df.astype(int)

            plt.clf()
            for index in F1_df.index:
                index_score = score_df.loc[index].values
                # save the stopping feature size by different methods
                index_all.append(tasks[i] + '_' + models[j] + '_' + index)
                plt.plot(range(1, len(index_score)+1), index_score,  linestyle='-', label=index)

            plt.xlabel('Number of Features')
            plt.ylabel('Test Score')
            # plt.xlim((1, 25))
            # plt.ylim((0.6, 1))
            plt.legend()
            plt.grid()
            if not os.path.exists('./Evaluation/KDDCUP99/F1_png/'):
                os.makedirs('./Evaluation/KDDCUP99/F1_png/')
            plt.savefig(os.path.join('./Evaluation/KDDCUP99/F1_png', F1_filename + '_Test.png'))
            
            Time_filename = tasks[i] + '_Time_' + models[j]
            Time_df = pd.read_csv('./Results/KDDCUP99/' + Time_filename + '.csv', index_col=0)
            time_df = pd.DataFrame(index=Time_df.index)

            for column in Time_df.columns:
                for index in Time_df.index:
                    time = float(Time_df.loc[index][column])
                    time_df.at[index, column] = time

            plt.clf()
            for index in Time_df.index:
                index_time = Time_df.loc[index].values

                # save the stopping feature size by different methods
                index_all.append(tasks[i] + '_' + models[j] + '_' + index)

                plt.plot(range(1, len(index_time)+1), index_time,  linestyle='-', label=index)

            plt.xlabel('Number of Features')
            plt.ylabel('Time')
            # plt.xlim((1, 25))
            # plt.ylim((0.6, 1))
            plt.legend()
            plt.grid()
            if not os.path.exists('./Evaluation/KDDCUP99/Time_png/'):
                os.makedirs('./Evaluation/KDDCUP99/Time_png/')
            plt.savefig(os.path.join('./Evaluation/KDDCUP99/Time_png', Time_filename + '.png'))


def text2tuple(text, task):
    info = text[1:-1].split(', ')
    if task == 'Set':
        return float(info[0]), float(info[1]), int(info[2])
    else:
        return float(info[0]), float(info[1])


if __name__ == "__main__":
    main()