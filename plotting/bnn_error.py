import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def plot(y_errors):
    # Number of data points
    n = 5

    # Dummy data
    np.random.seed(10)
    x = np.arange(0, n, 1)
    y = np.arange(0, n, 1)
    # y = np.random.rand(n) * 5.

    # Dummy errors (above and below)
    xerr = np.ones((2, n))*0.1 #np.random.rand(2, n) + 0.1
    yerr = np.ones((2, n))*0.2 #np.random.rand(2, n) + 0.2


    def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
                         edgecolor='None', alpha=1.0):
        # Create list for all the error patches
        errorboxes = []

        # Loop over data points; create box from errors at each point
        for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
            rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
            errorboxes.append(rect)

        # Create patch collection with specified colour/alpha
        pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                             edgecolor=edgecolor)

        # Add collection to axes
        ax.add_collection(pc)

        # Plot errorbars
        artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
                              fmt='None', ecolor=facecolor, alpha=alpha, elinewidth=0)

        return artists


    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Call function to create error boxes
    _ = make_error_boxes(ax, x, y, xerr, yerr)

    plt.show()


# load csv
# extract pre/post bnn errors
# set xerr to be static
# set yerr to be difference between pre and post
# set center as mean
# if starting value is less than previous post, color green
# otherwise color red

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
    header, rows = load_data("data.csv")
    cols = list(map(list, zip(*rows)))
    print (header)
    pre_bnn_errors = cols[-2]
    post_bnn_errors = cols[-1]
    # diff between post[i], pre[i+1]
    y_errors = [post_bnn_errors[i]-pre_bnn_errors[i+1] for i in range(len(pre_bnn_errors)-1)]
    # a postive y error -> increased
    # negative y error -> decreased
    plot(y_errors)
