"""Utility functions for setting up the experiments."""

import logging
import os
import sys

import yaml

from utils.auxiliary import make_sure_dir_exists


def setup(args, impute_args=True):
    """Sets up an experiment.

    Args:
        args: Command line arguments passed on experiment run.
        impute_args: `bool` indicating whether to impute the args' values,
            or not not.

    Returns:
        config: A dictionary of the experiment configurations.
        params: A dictionary of the model parameters.
    """
    # Setting up experiment directories
    dirs_config = setup_experiment_dirs(args.model_name)

    # Reading configs
    if args.reuse_config:
        config, params = load_configs(
            os.path.join(dirs_config['config_dir'], 'config.yaml'),
            os.path.join(dirs_config['config_dir'], 'params.yaml')
        )
    else:
        config, params = load_configs(args.config, args.params)

    config['dirs'] = dirs_config

    # Imputing passed values
    config['model_name'] = args.model_name
    if impute_args:
        config['training']['epochs'] = args.epochs if args.epochs else config['training']['epochs']
        config['training']['learning_rate'] = args.lr if args.lr else config['training']['learning_rate']

    # Saving updated configs
    save_configs(config, params, config['dirs']['config_dir'])

    return config, params


def setup_experiment_dirs(model_name, results_dir='../results'):
    """Sets up experiment directories.

    Args:
        model_name: A name of the model (experiment).
        results_dir: A directory for all experiment results.

    Returns:
        dirs_config: A dictionary of the directory configurations.
    """
    make_sure_dir_exists(results_dir)

    dirs_config = {'exp_dir': os.path.join(results_dir, model_name)}

    dirs_config['ckpt_dir'] = os.path.join(dirs_config['exp_dir'], 'ckpt')
    dirs_config['model_dir'] = os.path.join(dirs_config['ckpt_dir'], 'model')
    dirs_config['encoders_dir'] = os.path.join(dirs_config['ckpt_dir'], 'encoders')

    dirs_config['model_last_dir'] = os.path.join(dirs_config['model_dir'], 'last')
    dirs_config['encoders_last_dir'] = os.path.join(dirs_config['encoders_dir'], 'last')

    dirs_config['config_dir'] = os.path.join(dirs_config['exp_dir'], 'config')
    dirs_config['samples_dir'] = os.path.join(dirs_config['exp_dir'], 'samples')
    dirs_config['logs_dir'] = os.path.join(dirs_config['exp_dir'], 'logs')

    for path in dirs_config.values():
        make_sure_dir_exists(path)

    dirs_config['model_stats_file'] = os.path.join(dirs_config['model_dir'], 'steps')
    dirs_config['encoders_stats_file'] = os.path.join(dirs_config['encoders_dir'], 'steps')

    return dirs_config


def load_configs(config_file, params_file):
    """Loads configration files.

    Args:
        config_file: A path to the experiment configurations file.
        params_file: A path to the model parameters.

    Returns:
        config: A dictionary of the experiment configurations.
        params: A dictionary of the model parameters.
    """
    if not os.path.exists(config_file):
        raise ValueError('Incorrect config file path')

    if not os.path.exists(params_file):
        raise ValueError('Incorrect params file path')

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    return config, params


def save_configs(config, params, config_dir):
    """Saves configuration files.

    Args:
        config: A dictionary of the experiment configurations.
        params: A dictionary of the model parameters.
        config_dir: A directory for saving the files.
    """
    with open(os.path.join(config_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f)

    with open(os.path.join(config_dir, 'params.yaml'), 'w') as f:
        yaml.safe_dump(params, f)


def get_logger(name, logs_dir):
    """Returns the logger for the experiment.

    Args:
        name: A logger name
        logs_dir: A directory for saving the logs.

    Returns:
        logger: A logger object.
    """
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(os.path.join(logs_dir, f'{name}.log'))
    file_handler.setFormatter(formatter)

    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(screen_handler)

    return logger
