import os
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.interpolate import spline

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--load-path1', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--load-path2', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
args = parser.parse_args()

def parse_file(fpath):
    datas = []
    with open (fpath, 'r') as f:
        f.readline()
        for line in f:
            tmp = json.loads(line)
            t_time = float(tmp['t'])# + t_start
            # tmp = [t_time, int(tmp['l']), float(tmp['r'])]
            tmp = [t_time, float(tmp['r'])]
            # print (tmp)
            datas.append(tmp)
    return np.array(datas)


def plot(datas1, datas2):
    n = 3000

    # data1
    # get mean
    # print (datas[0][:,1].shape)
    # print (datas[1][:,1].shape)
    ydata1 = np.stack([data1[:,1][0:n] for data1 in datas1])
    print (ydata1.shape)
    ys1 = np.mean(ydata1, axis=0)
    ystd1 = np.std(ydata1, axis=0)
    print (ys1.shape)
    # ys = datas[:,1]
    ax = plt.gca()
    # ax.fill_between(xs, ys1-ystd1, ys1+ystd1, facecolor='green', alpha=0.5)
    # x_smooth = np.linspace(xs.min(), xs.max(), 100)
    # y_smooth = spline(xs, ys, x_smooth)
    plt.plot(np.arange(len(ys1)), ys1, color='blue', alpha=0.75)

    ydata2 = np.stack([data2[:,1][0:n] for data2 in datas2])
    print (ydata2.shape)
    ys2 = np.mean(ydata2, axis=0)
    ystd2 = np.std(ydata2, axis=0)
    print (ys2.shape)
    # ys = datas[:,1]
    ax = plt.gca()
    # ax.fill_between(xs, ys2-ystd2, ys2+ystd2, facecolor='red', alpha=0.5)
    # x_smooth = np.linspace(xs.min(), xs.max(), 100)
    # y_smooth = spline(xs, ys, x_smooth)
    plt.plot(np.arange(len(ys2)), ys2, color='orange', alpha=0.75)

    plt.show()

if __name__ == '__main__':
    """
    basically load several monitor files and plot mean, variance
    """
    data1 = []
    for i in range(1, 2):
        datas1 = parse_file(os.path.join(args.load_path1, str(i)+'/0.monitor.json'))
        data1.append(datas1)

    data2 = []
    for i in range(1, 2):
        datas2 = parse_file(os.path.join(args.load_path2, str(i)+'/0.monitor.json'))
        data2.append(datas2)
    plot(data1, data2)
