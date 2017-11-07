import argparse
from os import getenv
from os.path import join

import yaml


def common_parser():
    '''return the argument parser'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help='Output directory', required=True)
    parser.add_argument('-c', '--config',  default=None,
                        help='The projectionizer YAML config file. Must be passed when running the analysis for the first time. See templates/ folder for some examples')
    parser.add_argument('-l', '--no-local-scheduler',  action='store_true',
                        help='Do not use luigi\'s local scheduler (luigi daemon (luigid) must be running)')

    return parser


def luigi_params(config, args):
    '''Prepare the list of parameters to be passed to luigi.run command'''
    with open(config) as configfile:
        config_list = yaml.load(configfile)

    cmd = list()
    for task, options in config_list.items():
        for option, value in options.items():
            prefix = task + '-' if task != 'CommonParams' else ''
            cmd += ['--' + prefix + option.replace('_', '-'), str(value)]

    if not args.no_local_scheduler:
        cmd.append('--local-scheduler')

    return cmd
