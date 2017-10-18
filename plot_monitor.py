import argparse
import numpy as np
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--load-path', default='./trained_models/',
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
            print (tmp)
            datas.append(tmp)
    return datas


def plot(datas):
    datas = np.array(datas)
    # xs = datas[:,0]
    xs = np.arange(len(datas))
    ys = datas[:,1]
    plt.plot(xs, ys)
    plt.show()

if __name__ == '__main__':
    datas = parse_file(args.load_path)
    plot(datas)
