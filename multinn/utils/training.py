"""Utility functions for training and employing the models."""

import numpy as np
import tensorflow as tf
from six.moves import cPickle


class TrainingStats:
    """Training stats container."""

    def __init__(self, steps=0, epoch=0, run=0, metric_best=1e3):
        """Initialized the TrainingStats container.

        Args:
            steps: The number of training steps.
            epoch: The number of epochs.
            run: The training run number.
            metric_best: The best value for the monitored metric.
        """
        self._steps = steps
        self._epoch = epoch
        self._run = run
        self._metric_best = metric_best
        self._idle_epochs = 0

    @property
    def steps(self):
        """int: the number of passed training steps."""
        return self._steps

    @property
    def epoch(self):
        """int: the number of passed epochs."""
        return self._epoch

    @property
    def run(self):
        """int: the training run number."""
        return self._run

    @property
    def metric_best(self):
        """float: the best value for the monitore metric."""
        return self._metric_best

    @property
    def idle_epochs(self):
        """int: the number of epochs without improvement."""
        return self._idle_epochs

    def new_epoch(self):
        """Increases the `epoch` counter"""
        self._epoch += 1

    def new_step(self):
        """Increases the `steps` counter"""
        self._steps += 1

    def new_run(self):
        """Increases the `run` counter"""
        self._run += 1

    def update_metric_best(self, val):
        """Updates the best value for the monitored metric.

        Args:
            val: The new best metric value.
        """
        self._metric_best = val

    def new_idle_epoch(self):
        """Increases the `idle_epochs` counter"""
        self._idle_epochs += 1

    def reset_idle_epochs(self):
        """Resets the `idle_epochs` counter"""
        self._idle_epochs = 0

    def load(self, filename):
        """Loads the Training Stats from disk.

        Args:
            filename: The path to the file with stats.
        """
        with open(filename, 'rb') as f:
            self._steps, self._epoch, self._run, self._metric_best = cPickle.load(f)

    def save(self, filename):
        """Saves the Training Stats to disk.

        Args:
            filename: The path where to save the file with stats.
        """
        with open(filename, 'wb') as f:
            cPickle.dump((self.steps, self.epoch, self.run, self.metric_best), f)


class LossAccumulator:
    """Training loss accumulator."""

    def __init__(self):
        """Initializes the LossAccumulator."""
        self._sum_loss = 0.
        self._num_loss = 0
        self._num_nan = 0
        self._num_posinf = 0
        self._num_neginf = 0
        self._num_total = 0

    def clear(self):
        """Clears the data."""
        self.__init__()

    def update(self, loss):
        """Updates the data with new loss value.

        Args:
            loss: A new loss value.
        """
        self._num_total += 1

        if np.isnan(loss):
            self._num_nan += 1
        elif np.isposinf(loss):
            self._num_posinf += 1
        elif np.isneginf(loss):
            self._num_neginf += 1
        else:
            self._num_loss += 1
            self._sum_loss += loss

    def loss(self):
        """Returns the mean loss omitting bad values."""
        return self._sum_loss / self._num_loss if self._num_loss > 0 else np.nan

    def num_bad(self):
        """Returns the number of bad losses."""
        return self._num_nan + self._num_posinf + self._num_neginf

    def ratio_bad(self):
        """Returns the ratio of bad losses."""
        return self.num_bad() / self._num_total

    def __str__(self):
        """Returns a pretty str of the loss values."""
        return f' - loss: {self.loss():7.3f} ' \
               f'(nan: {self._num_nan}, +inf: {self._num_posinf}, -inf: {self._num_neginf}, ' \
               f'bad: {100. * self.ratio_bad():.2f}%)'


def compute_gradients(optimizer, loss, var_list):
    """Computes the gradients and returns the gradient update ops and summaries.

    Args:
        optimizer: A tf.Optimizer object for computing and applying the gradients.
        loss: A metric to minimize.
        var_list: A list of variables to optimize.

    Returns:
        update_ops: A list of gradient update ops.
        gradient_summary: A tf.Summary object for monitoring gradient values.
    """
    gradients, variables = zip(*optimizer.compute_gradients(loss, var_list))

    with tf.variable_scope('gradients/'):
        clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=5.)

        gradients = list(zip(clipped, variables))

        gradient_summary = tf.summary.merge(
            [tf.summary.histogram(var.name, grad) for grad, var in gradients]
        )

    with tf.name_scope('update_ops'):
        update_ops = optimizer.apply_gradients(gradients)

    return update_ops, gradient_summary


def collect_metrics(sess, metrics_upd, data, data_lengths, placeholders, batch_size, piece_size):
    """Collects the metrics over the dataset.

    Args:
        sess: A tf.Session instance for running the metrics update ops.
        metrics_upd: A list of metrics update ops.
        data: The data over which metrics are computed,
            sized `[num_eval, time_steps, num_dims, num_tracks]`.
        data_lengths: The lengths of the data, sized `[num_eval]`.
        placeholders: A dictionary of model's placeholders.
        batch_size: An evaluation batch size.
        piece_size: A length of sequences in time steps.
    """
    # Getting and zeroing out a list of streamable variables that store the metrics.
    stream_vars = [v for v in tf.local_variables() if 'metrics/' in v.name]
    sess.run(tf.variables_initializer(stream_vars))

    x = placeholders['x']
    lengths = placeholders['lengths']
    is_train = placeholders['is_train']

    # Batched evaluation of the data.
    for i in range(0, data.shape[0], batch_size):
        for j in range(0, data.shape[1], piece_size):
            seq_length_batch = data_lengths[i:i + batch_size]
            seq_length_batch = np.minimum(seq_length_batch, piece_size)

            max_length = seq_length_batch.max()
            songs_batch = data[i:i + batch_size, j:j + max_length, :]

            sess.run([metrics_upd],
                     feed_dict={x: songs_batch,
                                lengths: seq_length_batch,
                                is_train: False})


def generate_music(sess, sampler, intro_songs, placeholders, num_songs=5, concat=True):
    """Generates new music from the intro input samples.

    Args:
        sess: A tf.Session instance for running the metrics update ops.
        sampler: A sampling op that returns generated samples.
        intro_songs: An array of intro songs,
            sized `[num_sample, time_steps, num_dims, num_tracks]`.
        placeholders: A dictionary of model's placeholders.
        num_songs: A number of songs to generate for each intro input.
        concat: `bool` indicating whether to concatenate generated samples
            with the intro songs, or not.
    """
    x = placeholders['x']
    is_train = placeholders['is_train']

    intro = np.tile(intro_songs, (num_songs, 1, 1, 1))
    samples = sess.run(sampler, feed_dict={x: intro, is_train: False})

    if concat:
        music = np.concatenate([intro, samples], axis=1)
    else:
        music = samples

    return music
