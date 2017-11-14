# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from argparse import ArgumentParser
import os
import pandas as pd
from sklearn import model_selection

import featureselect
import models


def parse_args(inargs=None):
    """ Parses input arguments """
    parser = ArgumentParser("./loader.py")
    standard_path = '/Users/matthew/Github/CancerAnalysis/'

    iargs = parser.add_argument_group('Input Files/Data')
    iargs.add_argument('--csv_file',
                       default=os.path.join(standard_path, 'data.csv'),
                       help='Path to CSV File')

    if not inargs:
        args = parser.parse_args()
    else:
        args = parser.parse_args(inargs)
    return args


def split_data(data_df):
    """ Creates a input train, input test, output train, and output test
        data set where 'diagnosis' is the column for output
    """
    out_df = data_df['diagnosis']
    in_df = data_df.drop('diagnosis', axis=1)
    return model_selection.train_test_split(in_df, out_df, test_size=0.30)


def main(args):
    # Loads CSV File
    data_df = pd.read_csv(args.csv_file, index_col=0)
    data_df = data_df.iloc[:, :-1]

    # Cleans CSV File
    data_df['diagnosis'] = data_df['diagnosis'].astype('category')
    #data_df[data_df['diagnosis'] == 'M'] = 1
    #data_df[data_df['diagnosis'] == 'B'] = 0

    # Visualizes Data
    #featureselect.plot_features(data_df)

    # Splits CSV file
    in_train, in_test, out_train, out_test = split_data(data_df)
    lda_test_predict = models.lda(in_train, out_train, in_test, out_test)
    qda_test_predict = models.qda(in_train, out_train, in_test, out_test)

    return in_train, in_test, out_train, out_test






if __name__ == "__main__":
    ARGS = parse_args()
    in_train, in_test, out_train, out_test = main(ARGS)
    #models.random_forest(data_df)