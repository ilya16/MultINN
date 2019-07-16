"""Evaluating the MultINN models."""

import argparse
import os

import numpy as np
import tensorflow as tf

from models.multinn.multinn import MultINN
from utils.data import load_data, prepare_sampling_inputs
from utils.setup import get_logger, setup
from utils.training import TrainingStats, collect_metrics


def main(args):
    # Setting up an experiment
    config, params = setup(args, impute_args=False)

    # Setting up logger
    logger = get_logger(config['model_name'], config['dirs']['logs_dir'])

    # Extracting configurations
    data_config = config['data']
    logs_config = config['logs']
    training_config = config['training']
    sampling_config = config['sampling']
    dirs_config = config['dirs']
    logger.info('[SETUP] Experiment configurations')
    logger.info(f'[SETUP] Experiment directory: {os.path.abspath(dirs_config["exp_dir"])}')

    # Loading the dataset
    (X_train, len_train), (X_valid, len_valid), (X_test, len_test) = load_data(
        data_config=data_config,
        step_size=training_config['num_pixels']
    )

    # Computing beat size in time steps
    beat_size = float(data_config['beat_resolution'] / training_config['num_pixels'])

    # Creating the MultINN model
    tf.reset_default_graph()
    model = MultINN(config, params, mode=params['mode'], name=config['model_name'])
    logger.info('[BUILT] Model')

    # Extracting placeholders, metrics and summaries
    placeholders = model.placeholders

    metrics, metrics_upd, summaries = model.metrics, model.metrics_upd, model.summaries
    loglik = metrics['log_likelihood']

    metrics_sum = summaries['metrics']

    # TensorFlow Session set up
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf.set_random_seed(training_config['random_seed'])
    np.random.seed(training_config['random_seed'])

    with tf.Session(config=tf_config) as sess:
        logger.info('[START] TF Session')
        with tf.variable_scope('init_global'):
            init_global = tf.global_variables_initializer()
        with tf.variable_scope('init_local'):
            init_local = tf.local_variables_initializer()
        sess.run([init_global, init_local])

        stats = TrainingStats()

        # Loading the model's weights or using initial weights
        if args.from_last:
            if model.load(sess, dirs_config['model_last_dir']):
                last_stats_file = os.path.join(dirs_config['model_last_dir'], 'steps')
                if os.path.isfile(last_stats_file):
                    stats.load(last_stats_file)
                    logger.info('[LOAD]  Training stats file')

                logger.info(f'[LOAD]  Pre-trained weights (last, epoch={stats.epoch})')
            else:
                logger.info('[LOAD]  Initial weights')
        elif model.load(sess, dirs_config['model_dir']):
            if os.path.isfile(dirs_config['model_stats_file']):
                stats.load(dirs_config['model_stats_file'])
                logger.info('[LOAD]  Training file')

            logger.info(f'[LOAD]  Pre-trained weights (best, epoch={stats.epoch})')
        else:
            logger.error('[ERROR]  No checkpoint found')
            raise ValueError('No checkpoint found')

        # Preparing to the evaluation
        writer_train = tf.summary.FileWriter(f'{dirs_config["logs_dir"]}/Graph/run_{stats.run}/train')
        writer_valid = tf.summary.FileWriter(f'{dirs_config["logs_dir"]}/Graph/run_{stats.run}/valid')
        writer_test = tf.summary.FileWriter(f'{dirs_config["logs_dir"]}/Graph/run_{stats.run}/test')

        batch_size = training_config['batch_size']
        piece_size = int(training_config['piece_size'] * beat_size)

        num_eval = X_valid.shape[0]

        collect_metrics(sess, metrics_upd, data=X_train[:num_eval, ...], data_lengths=len_train[:num_eval, ...],
                        placeholders=placeholders, batch_size=batch_size * 2, piece_size=piece_size)
        summary, loglik_val = sess.run([metrics_sum, loglik])
        writer_train.add_summary(summary, stats.epoch)
        del summary
        logger.info(f'[EVAL]  Training   set log-likelihood:  {loglik_val:7.3f}')

        collect_metrics(sess, metrics_upd, data=X_valid, data_lengths=len_valid, placeholders=placeholders,
                        batch_size=batch_size * 2, piece_size=piece_size)
        summary, loglik_val = sess.run([metrics_sum, loglik])
        writer_valid.add_summary(summary, stats.epoch)
        del summary
        logger.info(f'[EVAL]  Validation set log-likelihood:  {loglik_val:7.3f}')

        collect_metrics(sess, metrics_upd, data=X_test[:num_eval, ...], data_lengths=len_test[:num_eval, ...],
                        placeholders=placeholders, batch_size=batch_size * 2, piece_size=piece_size)
        summary, loglik_val = sess.run([metrics_sum, loglik])
        writer_test.add_summary(summary, stats.epoch)
        del summary
        logger.info(f'[EVAL]  Test set       log-likelihood:  {loglik_val:7.3f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample music')

    parser.add_argument('--model-name', '-m', metavar='M', type=str,
                        help='the name of the model (experiment)',
                        default='MultINN')
    parser.add_argument('--config', '-c', metavar='C', type=str,
                        help='experiment configuration file path',
                        default='configs/test_config.yaml')
    parser.add_argument('--params', '-p', metavar='P', type=str,
                        help='model parameters file path',
                        default='configs/test_params.yaml')
    parser.add_argument('--reuse-config',
                        help='reuse saved configurations and ignore passed',
                        action='store_true')
    parser.add_argument('--from-last',
                        help='run the model with the last saved weights',
                        action='store_true')

    args = parser.parse_args()
    print(args)

    main(args)
