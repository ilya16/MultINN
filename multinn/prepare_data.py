"""Prepares the training data."""

import argparse
import os

import numpy as np

from utils.auxiliary import make_sure_dir_exists


def load_data_from_npz(filename, sample_size):
    """Loads the training data from a npz file (sparse format).

    Args:
        filename: A path to the data file.
        sample_size: A sample size (number of observations to use)

    Returns:
        data: Loaded data, sized `[num_samples, time_steps, pitch_span, num_tracks]`.
    """
    with np.load(filename) as f:
        shape = f['shape']
        shape[0] = sample_size

        mask = np.where(f['nonzero'][0] < sample_size)[0]
        nonzero = f['nonzero'][:, mask]

        data = np.zeros(shape, np.bool_)
        data[[x for x in nonzero]] = True
        del nonzero

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')

    parser.add_argument('--data-file', '-d', metavar='D', type=str,
                        help='dataset file', default='../data')
    parser.add_argument('--sample-size', '-s', metavar='S', type=int,
                        help='number of samples to use', default=40000)

    args = parser.parse_args()

    raw_data_file = args.data_file
    print(f'Reading data from {raw_data_file}')
    data = load_data_from_npz(
        raw_data_file,
        sample_size=args.sample_size
    )

    data_file = os.path.join(args.data_dir, 'X_train_all.npy')
    print(f'Saving as {data_file}')
    np.save(
        data_file,
        data.reshape((args.sample_size, -1, 84, 5))
    )
