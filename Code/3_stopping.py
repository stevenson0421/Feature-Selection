import argparse
import numpy as np
from numpy import random

import matplotlib.pyplot as plt



def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--method", default='MD', help="e.g. MD(MaxDelta), MPR(MinPerfReq), MS(MaxScore)", type=str)
    parser.add_argument("-t", "--tolerence", default=0.03, help="Tolerence rate for MPR.", type=float)
    parser.add_argument("-r", "--rho", default=0.003, help="Penalty coefficient for MS.", type=float)
    args = parser.parse_args()

    score_his = np.array([i for i in range(61, 100, 1)])
    feature_his = []

    np.random.seed(0)
    x = random.rand(len(score_his))*10
    score_his = score_his+x
    plt.grid()
    plt.plot(score_his)
    plt.savefig('tmp.png')


    # Get stopping points by the following methods.
    stopping_point = MaxDelta(score_his, args)
    print('MaxDelta', '\t', stopping_point)

    stopping_point = MinPerfReq(score_his, args)
    print('MinPerfReq', '\t', stopping_point)

    stopping_point = MaxScore(score_his, args)
    print('MaxScore', '\t', stopping_point)


def MaxDelta(score, args):
    max_delta = 0
    feature_size = len(score)
    for i in range(len(score)-1, 0, -1):
        delta = score[i] - score[i-1]
        if delta > max_delta:
            max_delta = delta
            feature_size = i + 1
            
    return feature_size

def MinPerfReq(score, args):
    best_CVscore = score[-1]
    feature_size = len(score)
    for i in range(len(score)-1, -1, -1):
        delta = best_CVscore - score[i]
        if delta > args.tolerence:
            feature_size = i + 1
            break

    return feature_size

def MaxScore(score, args):
    best_performance = 0
    feature_size = len(score)
    for i in range(len(score)-1, -1, -1):
        current_size = i + 1
        adj_score = score[i] - (args.rho * current_size)
        if adj_score > best_performance:
            best_performance = adj_score
            feature_size = i + 1

    return feature_size


if __name__ == "__main__":
    main()