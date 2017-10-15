"""
Some utils for saving useful data
"""

import joblib
import os
import csv

class Logger(object):

    def __init__(self, args):
        self.args = args
        self.csvfile = None
        self.csvwriter = None

    def save_args(self):
        # save args for reference
        joblib.dump(self.args, os.path.join(self.args.log_dir, 'args_snapshot.pkl'))

    def create_csv_log(self):
        # setup csv logging
        self.csvfile = open(os.path.join(self.args.log_dir, 'data.csv'), 'w')
        fields = ['updates',
                  'frames',
                  'mean_reward',
                  'median_reward',
                  'min_reward',
                  'max_reward',
                  'pol_entropy',
                  'value_loss',
                  'policy_loss',
                  'raw_kls',
                  'scaled_kls',
                  'bonuses',
                  'replay_size',
                  'latest_pre_bnn_error',
                  'latest_post_bnn_error']
        self.csvwriter = csv.DictWriter(self.csvfile, fieldnames=fields)
        self.csvwriter.writeheader()
        self.csvfile.flush()

    def write_row(self, row_dict):
        self.csvwriter.writerow(row_dict)
        self.csvfile.flush()
