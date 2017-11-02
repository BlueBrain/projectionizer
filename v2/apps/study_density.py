#!/usr/bin/env python
import argparse
import json
import logging
import os

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from examples.decorators import load_feather
from examples.layer_data import all_data

# layer_data = all_data(json.load(open('exp_data.json')))


L = logging.getLogger(__name__)


def density(df):
    plt.figure()
    mult = 500000

    def plot_density(layer, norm):
        densities_data = layer_data[layer]['density']
        y, densities = zip(*densities_data)
        plt.plot(y, np.array(densities) * norm, 'o-')

    plt.hist(df['y'], np.linspace(600, 1600, 200))
    # plot_density(4, mult)
    # plot_density(6, mult)


def connection_per_synapse(df):
    plt.figure()
    bins = np.arange(50)
    g = df.groupby(['sgid', 'tgid'], sort=False).size()
    plt.hist(g, bins=bins)

    plt.plot(bins, mlab.normpdf(bins, 7, 1) * g.sum() / np.sqrt(2 * np.pi))


def get_parser():
    '''return the argument parser'''
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input file')

    parser.add_argument('-o', '--output', default='.',
                        help='Output directory')
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0,
                        help='-v for INFO, -vv for DEBUG')
    return parser


def main(args):
    logging.basicConfig(level=(logging.WARNING,
                               logging.INFO,
                               logging.DEBUG)[min(args.verbose, 2)])

    df = pd.read_feather(os.path.expanduser(args.input))

    density(df)
    # connection_per_synapse(df)
    plt.show()


if __name__ == '__main__':
    PARSER = get_parser()
    main(PARSER.parse_args())
