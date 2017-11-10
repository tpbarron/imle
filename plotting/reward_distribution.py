import os
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.interpolate import spline
plt.rc('font', family='serif')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--load-path1', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--load-path2', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--load-path3', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--n', type=int, default=250000)
parser.add_argument('--label1', type=str, default='label1')
parser.add_argument('--label2', type=str, default='label2')
parser.add_argument('--label3', type=str, default='label3')
parser.add_argument('--name', type=str, default='name')
args = parser.parse_args()

# use window = 10000 for mtncar
# 50000 for walker
# 8000 for acrobot
#
def smooth(x,window_len=8000,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError ("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def parse_file(fpath):
    datas = []
    with open (fpath, 'r') as f:
        f.readline()
        for line in f:
            tmp = json.loads(line)
            t_time = float(tmp['t'])# + t_start
            # tmp = [t_time, int(tmp['l']), float(tmp['r'])]
            tmp = [t_time, float(tmp['r']), int(tmp['l'])]
            # print (tmp)
            datas.append(tmp)
    # datas is time, rew, ep steps
    return np.array(datas)

def make_y_given_cumsum(cumsum, y, n=250000):
    newy = np.zeros((n,))
    for i in range(len(cumsum)-1):
        newy[int(cumsum[i]):] = y[i]
    return newy

def plot(datas1, datas2, datas3):
    n = args.n
    # n = 3000
    S = 10

    fig = plt.figure(figsize=(4, 3))

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # data1
    # get mean
    # print (datas[0][:,1].shape)
    # print (datas[1][:,1].shape)
    print ([data1[:,1].shape for data1 in datas1])

    xs = np.arange(n)
    ylist = [make_y_given_cumsum(np.cumsum(data1[:,2]), data1[:,1], n) for data1 in datas1]
    # print ([l.shape for l in ylist])
    ydata1 = np.stack(ylist)
    # ydata1 = np.stack([data1[:,1] for data1 in datas1])
    # print (ydata1.shape)
    ys1 = np.mean(ydata1, axis=0)
    ys1 = smooth(ys1)[:n]
    # ys1 = np.convolve(ys1, np.ones((S,))/S, mode='same')
    ystd1 = np.std(ydata1, axis=0)
    # print (ys1.shape)
    # ys = datas[:,1]
    ax = plt.gca()
    # ax.fill_between(xs, ys1-ystd1, ys1+ystd1, facecolor='blue', alpha=0.25)
    # x_smooth = np.linspace(xs.min(), xs.max(), 100)
    # y_smooth = spline(xs, ys, x_smooth)
    # print (datas1[0][:,2][0:n].shape)
    # xs = np.cumsum(datas1[0][:,2])
    # print (xs.shape, ys1.shape)
    plt.plot(xs, ys1, color='blue', alpha=0.75, label=args.label1)
    # plt.plot(np.arange(len(ys1)), ys1, color='blue', alpha=0.75)

    xs2 = np.arange(n)
    ylist = [make_y_given_cumsum(np.cumsum(data2[:,2]), data2[:,1], n) for data2 in datas2]
    # print ([l.shape for l in ylist])
    ydata2 = np.stack(ylist)
    # ydata2 = np.stack([data2[:,1] for data2 in datas2])
    # print (ydata2.shape)
    ys2 = np.mean(ydata2, axis=0)
    ys2 = smooth(ys2)[:n]
    # ys2 = np.convolve(ys2, np.ones((S,))/S, mode='same')
    ystd2 = np.std(ydata2, axis=0)
    # print (ys2.shape)
    # ys = datas[:,1]
    ax = plt.gca()
    # ax.fill_between(xs2, ys2-ystd2, ys2+ystd2, facecolor='orange', alpha=0.25)
    # x_smooth = np.linspace(xs.min(), xs.max(), 100)
    # y_smooth = spline(xs, ys, x_smooth)
    # xs2 = np.cumsum(datas2[0][:,2])
    plt.plot(xs2, ys2, color='orange', alpha=0.75, label=args.label2)

    xs3 = np.arange(n)
    ylist = [make_y_given_cumsum(np.cumsum(data3[:,2]), data3[:,1], n) for data3 in datas3]
    # print ([l.shape for l in ylist])
    ydata3 = np.stack(ylist)
    # ydata2 = np.stack([data2[:,1] for data2 in datas2])
    # print (ydata2.shape)
    ys3 = np.mean(ydata3, axis=0)
    ys3 = smooth(ys3)[:n]
    # ys2 = np.convolve(ys2, np.ones((S,))/S, mode='same')
    ystd3 = np.std(ydata3, axis=0)
    # print (ys2.shape)
    # ys = datas[:,1]
    ax = plt.gca()
    # ax.fill_between(xs3, ys3-ystd3, ys3+ystd3, facecolor='green', alpha=0.25)
    # x_smooth = np.linspace(xs.min(), xs.max(), 100)
    # y_smooth = spline(xs, ys, x_smooth)
    # xs2 = np.cumsum(datas2[0][:,2])
    plt.plot(xs3, ys3, color='green', alpha=0.75, label=args.label3)

    ax.set_xlabel('Steps')
    ax.set_ylabel('Reward')

    plt.legend()
    plt.tight_layout()

    plt.savefig('plotting/'+args.name+'.pdf', format='pdf')
    # plt.show()

if __name__ == '__main__':
    """
    basically load several monitor files and plot mean, variance
    """
    data1 = []
    for i in range(1, 4):
        datas1 = parse_file(os.path.join(args.load_path1, str(i)+'/0.monitor.json'))
        data1.append(datas1)

    data2 = []
    for i in range(1, 4):
        datas2 = parse_file(os.path.join(args.load_path2, str(i)+'/0.monitor.json'))
        data2.append(datas2)

    data3 = []
    for i in range(1, 4):
        datas3 = parse_file(os.path.join(args.load_path3, str(i)+'/0.monitor.json'))
        data3.append(datas3)

    plot(data1, data2, data3)
