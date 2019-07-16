"""Sampling using the MultINN models."""

import argparse
import os

import numpy as np
import tensorflow as tf

from metrics.musical import compute_sample_metrics
from models.multinn.multinn import MultINN
from utils.data import load_data, prepare_sampling_inputs, pad_to_midi, save_music
from utils.setup import get_logger, setup
from utils.training import TrainingStats, generate_music


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
    (X_train, len_train), (X_valid, len_valid), (_, _) = load_data(
        data_config=data_config,
        step_size=training_config['num_pixels']
    )

    # Computing beat size in time steps
    beat_size = float(data_config['beat_resolution'] / training_config['num_pixels'])

    # Preparing inputs for sampling
    intro_songs, save_ids, song_labels = prepare_sampling_inputs(X_train, X_valid, sampling_config, beat_size)
    num_save_intro = len(save_ids) // sampling_config['num_save']
    logger.info('[SETUP] Inputs for sampling')

    # Creating the MultINN model
    tf.reset_default_graph()
    model = MultINN(config, params, mode=params['mode'], name=config['model_name'])
    logger.info('[BUILT] Model')

    # Building the sampler and evaluator
    sampler = model.sampler(num_beats=sampling_config['sample_beats'])
    logger.info('[BUILT] Sampler')

    # Extracting placeholders, metrics and summaries
    placeholders = model.placeholders

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

        samples = generate_music(sess, sampler, intro_songs, placeholders,
                                 num_songs=sampling_config['num_songs'])
        logger.info('[EVAL]  Generated music samples')

        samples = pad_to_midi(samples, data_config)
        samples_to_save = samples[save_ids]

        # Saving the music
        if logs_config['save_samples_epochs'] > 0 and stats.epoch % logs_config['save_samples_epochs'] == 0:
            save_music(samples_to_save, num_intro=num_save_intro, data_config=data_config,
                       base_path=f'eval_{model.name}_e{stats.epoch}', save_dir=dirs_config['samples_dir'],
                       song_labels=song_labels)
            logger.info('[SAVE]  Saved music samples')

        if args.eval_samples:
            logger.info('[EVAL]  Evaluating music samples')
            samples = np.reshape(
                samples,
                (samples.shape[0], -1, data_config['beat_resolution'] * 4,) + samples.shape[-2:]
            )
            compute_sample_metrics(samples)


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
    parser.add_argument('--eval-samples',
                        help='evaluate generated sample using musical metrics',
                        action='store_true')

    args = parser.parse_args()
    print(args)

    main(args)
