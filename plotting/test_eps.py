import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import argparse

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--load-path1', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--load-path2', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
args = parser.parse_args()

import csv

def load_data(fname):
    header = None
    rows = []
    with open(fname, 'r') as f:
        csvreader = csv.reader(f)
        i = 0
        for row in csvreader:
            if i == 0:
                header = row
            else:
                rows.append(row)
            i += 1
    return header, rows

if __name__ == '__main__':
    header, rows = load_data(args.load_path1)
    cols = list(map(list, zip(*rows)))
    print (header)
    mean_rewards = []
    for x in cols[3]:
        mean_rewards.append(float(x))
    plt.plot(np.arange(len(mean_rewards)), np.array(mean_rewards), color='blue')

    header, rows = load_data(args.load_path2)
    cols = list(map(list, zip(*rows)))
    print (header)
    mean_rewards = []
    for x in cols[3]:
        mean_rewards.append(float(x))
    plt.plot(np.arange(len(mean_rewards)), np.array(mean_rewards), color='orange')

    plt.show()

    # post_bnn_errors = [float(x) for x in cols[-1]]
    # # print ('pre: ', pre_bnn_errors)
    # # print ('post: ', post_bnn_errors)
    # # diff between post[i], pre[i+1]
    # y_errors = [pre_bnn_errors[i+1]-post_bnn_errors[i] for i in range(len(pre_bnn_errors)-1)]
    # # a postive y error -> increased error
    # # negative y error -> decreased error
    # # print (pre_bnn_errors)
    # # print (post_bnn_errors)
    # # print (y_errors)
    # plot(pre_bnn_errors, post_bnn_errors, y_errors)
