import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

# def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='r',
#                      edgecolor='None', alpha=1.0):
#     # Create list for all the error patches
#     errorboxes = []
#
#     # Loop over data points; create box from errors at each point
#     for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
#         rect = Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum())
#         errorboxes.append(rect)
#
#     # Create patch collection with specified colour/alpha
#     pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
#                          edgecolor=edgecolor)
#
#     # Add collection to axes
#     ax.add_collection(pc)
#
#     # Plot errorbars
#     artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
#                           fmt='None', ecolor=facecolor, alpha=alpha, elinewidth=0)
#
#     return artists

def make_error_box(ax, x, y, w, h):
    rect = Rectangle((x - w/2., y - abs(h)/2.), w, abs(h))

    # Create patch collection with specified colour/alpha
    if h <= 0:
        fc = 'blue'
    else:
        fc = 'orange'
    pc = PatchCollection([rect], facecolor=fc, alpha=1.0,
                         edgecolor='None')

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    artists = ax.errorbar([x], [y], xerr=[w/2.], yerr=[h/2.],
                          fmt='None', ecolor=fc, alpha=1.0, elinewidth=0)

    return artists

def plot(pre_errors, post_errors, y_errors):
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # # Call function to create error boxes
    # _ = make_error_boxes(ax, x, y, xerr, yerr)

    for i in range(len(pre_errors)-1):
        post_error = post_errors[i]
        pre_error = pre_errors[i+1]
        y_err = y_errors[i]
        make_error_box(ax, i, (pre_error+post_error)/2., 1, y_err)

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
    pre_bnn_errors = [float(x) for x in cols[-2]]
    post_bnn_errors = [float(x) for x in cols[-1]]
    # print ('pre: ', pre_bnn_errors)
    # print ('post: ', post_bnn_errors)
    # diff between post[i], pre[i+1]
    y_errors = [pre_bnn_errors[i+1]-post_bnn_errors[i] for i in range(len(pre_bnn_errors)-1)]
    # a postive y error -> increased error
    # negative y error -> decreased error
    print (pre_bnn_errors)
    print (post_bnn_errors)
    print (y_errors)
    plot(pre_bnn_errors, post_bnn_errors, y_errors)
