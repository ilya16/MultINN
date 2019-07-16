"""Implementation of a RNN-MultiNADE generator."""

import itertools

import tensorflow as tf
from tensorflow.python.layers import core as tf_layers_core

from models.common.nade import NADE
from models.generators.rnn_estimator import RnnEstimatorStateTuple, RnnEstimator
from models.generators.rnn_nade import RnnNade
from utils.auxiliary import get_current_scope
from utils.sequences import flatten_maybe_padded_sequences


class RnnMultiNADE(RnnNade):
    """RNN-MultiNADE generator.

    The model with a shared temporal RNN cell and multiple NADEs,
    one per input sequence (track).

    Can be used to model the distributions of inputs for multiple
    parallel sequences (tracks) with the inter-track relations.

    The RNN-MultiNADE inherits the RNN-NADE interface.

    The model is utilized for the Composer MultINN models.
    """

    def __init__(self,
                 num_dims,
                 num_hidden,
                 num_hidden_rnn,
                 tracks,
                 keep_prob=1.0,
                 internal_bias=False,
                 name='rnn-multinade'):
        """Initializes the RNN-NADE model.

        Args:
            num_dims: The number of output dimensions of the model for a single track.
            num_hidden: The number of hidden units of each NADE.
            num_hidden_rnn: The number of hidden units of the temporal RNN cell.
            keep_prob: The portion of outputs to use during the training,
                inverse of `dropout_rate`.
            internal_bias: `bool` indicating whether the Generator should maintain
                its own bias variables. If `False`, external values must be passed.
            name: The model name to use as a prefix when adding operations.
        """

        self._tracks = tracks

        super().__init__(num_dims=num_dims, num_hidden=num_hidden, num_hidden_rnn=num_hidden_rnn,
                         keep_prob=keep_prob, internal_bias=internal_bias,
                         name=name, track_name='all')

        self._num_output = self.num_tracks * self.num_dims

    def _init_estimator(self):
        """Initializes the NADE Estimators and the Dense layer that connects RNN and NADEs."""
        self._fc_layer = tf_layers_core.Dense(
            units=self.num_tracks * (self.num_dims + self.num_hidden[-1]),
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )

        self._nades = []
        for i in range(self.num_tracks):
            with tf.name_scope(f'{get_current_scope()}{self.tracks[i]}/'):
                self._nades.append(
                    NADE(self.num_dims, self.num_hidden[-1],
                         internal_bias=self.internal_bias)
                )

    @property
    def tracks(self):
        """list of str: The names of tracks with which Estimators operate."""
        return self._tracks

    @property
    def num_tracks(self):
        """int: The number of tracks with which Estimators operate."""
        return len(self._tracks)

    def build(self, x=None, y=None, lengths=None, is_train=None, mode='eval'):
        """Builds RNN-MultiNADE's graph.

        Defines the forward pass through the input sequences.

        Note:
            The inputs to the model are zero padded for the first time step.

        Args:
            x: A batch of input sequences to the generator,
                sized `[batch_size, time_steps, ...]`.
            y: A batch of target predictions for the generator,
                sized `[batch_size, time_steps, num_dims]`.
            lengths: The lengths of input sequences sized `[batch_size]`.
            is_train: `bool` for distinguishing between training and inference.
            mode: Build mode for optimizing the graph size.
                Some operations may not be needed for training or inference.

        Raises:
            ValueError: If incorrect build mode is provided.
        """
        RnnEstimator.build(self, x, y, lengths, is_train, mode)

        if mode in ('train', 'eval'):
            inputs, targets = x, y

            # Flatten targets into the shape `[sum(lengths), num_dims]`
            # and split into a list of track targets
            with tf.variable_scope(f'{get_current_scope()}generator_targets/flat/'):
                targets_flat = tf.reshape(
                    flatten_maybe_padded_sequences(targets, lengths),
                    [-1, self.num_dims, self.num_tracks]
                )
                targets_flat = tf.unstack(targets_flat, axis=-1)

            # Compute log probabilities for inputs and targets by scanning through inputs
            with tf.name_scope(f'{get_current_scope()}{self.name}/'):
                log_probs, cond_probs = self.log_prob(inputs, targets_flat, lengths)

            # Model outputs binarization
            with tf.variable_scope(f'{get_current_scope()}generator_outputs/'):
                self._outputs = []

                for i in range(self.num_tracks):
                    with tf.variable_scope(self.tracks[i]):
                        self._outputs.append(tf.cast(tf.greater_equal(cond_probs[i], .5), tf.float32))

            # Building model metrics and summaries from the targets and outputs
            self._metrics, self._metrics_upd, self._summaries = self.build_metrics(
                targets=targets_flat,
                predictions=self._outputs,
                cond_probs=cond_probs,
                log_probs=log_probs
            )

        # RNN-MultiNADE variables
        self._variables = {
            'rnn': self._rnn.variables,
            'nade': [nade.variables for nade in self._nades],
            'dense': self._fc_layer.trainable_variables
        }

        self._trainable_variables = \
            self._rnn.trainable_variables \
            + list(itertools.chain.from_iterable([nade.trainable_variables for nade in self._nades])) \
            + self._fc_layer.trainable_variables

        self._summaries['weights'] = self.weight_summary

        self._is_built = True

    def build_metrics(self, targets, predictions, cond_probs=None, log_probs=None):
        """Builds RNN-MultiNADE metrics ops based on predictions and target values.

        Uses internal NADEs to compute the metrics for each track
        and averages the metrics to compute the metrics for the entire RNN-MultiNADE.

        Args:
            targets: A list of flattened target predictions for all tracks,
                sized `[batch_size, num_dims]`.
            predictions: A list of flattened model's predictions for all tracks,
                sized `[batch_size, num_dims]`.
            cond_probs: A list of flattened conditional probabilities for predictions.
                sized `[batch_size, num_dims]`.
            log_probs: A list of flattened log probabilities for predictions,
                sized `[batch_size]`.

        Returns:
            metrics: A dictionary of metric names and metric ops.
            metrics_upd: A list of metric update ops.
            metric_summaries: A tf.Summary object for model's metrics.

        Raises:
            ValueError: If none of cond_probs and log_probs is provided.
        """
        metrics = {}
        metrics_upd = []
        summaries = {'metrics': []}

        # Building metrics and summaries for each track
        for i in range(self.num_tracks):
            with tf.variable_scope(f"metrics/{self.name}/{self.tracks[i]}/"):
                metrics_i, metrics_upd_i, summaries_i = self._nades[i].build_metrics(
                    targets[i], predictions[i], cond_probs[i], log_probs[i]
                )

            for k, m in metrics_i.items():
                if k in metrics:
                    metrics[k].append(m)
                else:
                    metrics[k] = [m]

            metrics_upd += metrics_upd_i

            summaries['metrics'].append(summaries_i)

        # Metrics and summaries for the entire model
        with tf.variable_scope(f"metrics/{self.name}/global/"):
            # Averaging metrics
            for k in metrics.keys():
                metrics[k] = tf.reduce_mean(metrics[k], name=k)

            summary_global = [tf.summary.scalar(k, m) for k, m in metrics.items() if k != 'batch/loss']
            summaries['metrics'].append(tf.summary.merge(summary_global))

        summaries['metrics'] = tf.summary.merge(summaries['metrics'])

        return metrics, metrics_upd, summaries

    def zero_state(self, batch_size):
        """Creates the initial RNN-MultiNADE state, an RnnEstimatorStateTuple of zeros.

        Args:
            batch_size: batch size.

        Returns:
             An RnnEstimatorStateTuple of zeros.
        """
        with tf.variable_scope(f'{get_current_scope()}RnnRbmZeroState/', values=[batch_size]):
            initial_rnn_state = self._get_rnn_zero_state(batch_size)
            return RnnEstimatorStateTuple(
                [tf.zeros((batch_size, self._nade.num_hidden), name='b_enc')
                 for _ in range(self.num_tracks)],
                [tf.zeros((batch_size, self.num_dims), name='b_dec')
                 for _ in range(self.num_tracks)],
                initial_rnn_state
            )

    def _build_biases(self, outputs):
        """Builds NADE biases for `num_tracks` from the Dense layer outputs.

        Args:
            outputs: Dense layer outputs,
                sized `[batch_size, num_tracks x (num_dims + num_hidden[-1])]`.

        Returns:
            b_enc: A list of encoder bias terms, sized `[batch_size, num_hidden[-1]]`.
            b_dec: A list of decoder bias terms, sized `[batch_size, num_dims]`.
        """
        b_enc, b_dec = tf.split(
            outputs,
            [self.num_tracks * self.num_hidden[-1], self.num_tracks * self.num_dims],
            axis=1
        )

        b_enc = tf.split(b_enc, self.num_tracks, axis=1)
        b_dec = tf.split(b_dec, self.num_tracks, axis=1)

        if self.internal_bias:
            for i in range(self.num_tracks):
                b_enc[i] += self._nades[i].b_enc
                b_dec[i] += self._nades[i].b_dec

        return b_enc, b_dec

    def log_prob(self, inputs, targets_flat, lengths=None):
        """Computes the log probability of a sequence of values.

        Args:
            inputs: A batch of sequences to compute the log probabilities of,
                sized `[batch_size, time_steps, num_tracks x num_dims]`.
            targets_flat: A list The flattened target predictions,
                sized `[sum(lengths), num_dims]`
            lengths: The length of each sequence, sized `[batch_size]`
                or None if all are equal.

        Returns:
            log_prob: A list of the log probabilities for each track of the sequences,
                sized `[sum(lengths), 1]`.
            cond_prob: A list of the conditional probabilities for targets of each track,
                sized `[sum(lengths), num_dims]`.
        """
        assert self.num_tracks * self.num_dims == inputs.shape[2].value

        state = self._get_state(inputs, lengths=lengths)

        log_prob, cond_prob = [], []
        # Computing log probabilities for each track
        for i in range(self.num_tracks):
            with tf.variable_scope(f'log_probs/{self.tracks[i]}'):
                log_p, cond_p = self._nades[i].log_prob(
                    targets_flat[i], state.b_enc[i], state.b_dec[i]
                )

            log_prob.append(log_p)
            cond_prob.append(cond_p)

        return log_prob, cond_prob

    def sample_single(self, inputs, state):
        """Computes a sample and its probability from a batch of states.

        Args:
            inputs: (Unused) A batch of inputs from which sampling may start.
            state: An RnnEstimatorStateTuple containing the RNN-MultiNADE state
                for each sample in the batch

        Returns:
            eval: A sample for each input state, sized `[batch_size, num_tracks x num_dims]`.
            log_prob: A list of log probabilities of each track in each sample,
                sized `[batch_size, 1]`.
        """
        sample, log_prob = [], []

        for i in range(self.num_tracks):
            b_enc, b_dec = state.b_enc[i], state.b_dec[i]

            sample_i, log_prob_i = self._nades[i].sample(b_enc, b_dec, temperature=1.)
            sample.append(sample_i)
            log_prob.append(log_prob_i)

        sample = tf.stack(sample, axis=2)
        sample = tf.reshape(sample, [tf.shape(sample)[0], self._num_output])

        return sample, log_prob
