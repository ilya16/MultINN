"""Implementation of a RNN-NADE generator.

Note:
    Adapted from
    https://github.com/tensorflow/magenta/blob/master/magenta/models/pianoroll_rnn_nade/pianoroll_rnn_nade_graph.py
"""

import tensorflow as tf
from tensorflow.python.layers import core as tf_layers_core

from models.common.nade import NADE
from models.generators.rnn_estimator import RnnEstimator, RnnEstimatorStateTuple
from utils.auxiliary import get_current_scope
from utils.sequences import flatten_maybe_padded_sequences


class RnnNade(RnnEstimator):
    """RNN-NADE generator [1].

    [1]: http://www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf
    """

    def __init__(self,
                 num_dims,
                 num_hidden,
                 num_hidden_rnn,
                 keep_prob=1.0,
                 internal_bias=False,
                 name='rnn-nade',
                 track_name='all'):
        """Initializes the RNN-NADE model.

        Args:
            num_dims: The number of output dimensions of the model.
            num_hidden: The number of hidden units of the NADE.
            num_hidden_rnn: The number of hidden units of the temporal RNN cell.
            keep_prob: The portion of outputs to use during the training,
                inverse of `dropout_rate`.
            internal_bias: `bool` indicating whether the Generator should maintain
                its own bias variables. If `False`, external values must be passed.
            name: The model name to use as a prefix when adding operations.
            track_name: The track name on which encoder operates.
        """
        super().__init__(
            num_dims=num_dims, num_hidden=num_hidden, num_hidden_rnn=num_hidden_rnn,
            keep_prob=keep_prob, internal_bias=internal_bias,
            name=name, track_name=track_name
        )

        self._num_output = self.num_dims

    def _init_estimator(self):
        """Initializes the NADE Estimator and the Dense layer that connects RNN and NADE."""
        self._fc_layer = tf_layers_core.Dense(
            units=self.num_dims + self.num_hidden[-1],
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )

        self._nade = NADE(
            self.num_dims, self.num_hidden[-1],
            internal_bias=self.internal_bias
        )

    def build(self, x=None, y=None, lengths=None, is_train=None, mode='eval'):
        """Builds RNN-NADE's graph.

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
        super().build(x, y, lengths, is_train, mode)

        if mode in ('train', 'eval'):
            inputs, targets = x, y

            # Flatten targets into the shape `[sum(lengths), num_dims]`
            with tf.variable_scope(f'{get_current_scope()}generator_targets/{self.track_name}/flat/'):
                targets_flat = flatten_maybe_padded_sequences(targets, lengths)

            # Compute log probabilities for inputs and targets by scanning through inputs
            with tf.name_scope(f'{get_current_scope()}{self.name}/'):
                log_probs, cond_probs = self.log_prob(inputs, targets_flat, lengths)

            # Model outputs binarization
            with tf.variable_scope(f'{get_current_scope()}generator_outputs/{self.track_name}/'):
                self._outputs = tf.cast(tf.greater_equal(cond_probs, .5), tf.float32)

            # Building model metrics and summaries from the targets and outputs
            self._metrics, self._metrics_upd, self._summaries['metrics'] = self.build_metrics(
                targets=targets_flat,
                predictions=self._outputs,
                cond_probs=cond_probs,
                log_probs=log_probs
            )

        # RNN-NADE variables
        self._variables = {
            'rnn': self._rnn.variables,
            'nade': self._nade.variables,
            'dense': self._fc_layer.trainable_variables
        }

        self._trainable_variables = \
            self._rnn.trainable_variables \
            + self._nade.trainable_variables \
            + self._fc_layer.trainable_variables

        self._summaries['weights'] = self.weight_summary

        self._is_built = True

    def build_metrics(self, targets, predictions, cond_probs=None, log_probs=None):
        """Builds RNN-NADE metrics ops based on predictions and target values.

        Uses internal NADE to compute the metrics.

        Args:
            targets: Flattened target predictions,
                sized `[batch_size, num_dims]`.
            predictions: Flattened model's predictions,
                sized `[batch_size, num_dims]`.
            cond_probs: Flattened conditional probabilities for predictions.
                sized `[batch_size, num_dims]`.
            log_probs: Flattened log probabilities for predictions,
                sized `[batch_size]`.

        Returns:
            metrics: A dictionary of metric names and metric ops.
            metrics_upd: A list of metric update ops.
            metric_summaries: A tf.Summary object for model's metrics.

        Raises:
            ValueError: If none of cond_probs and log_probs is provided.
        """
        with tf.variable_scope(f"metrics/{self.name}/{self.track_name}/"):
            metrics, metrics_upd, summaries = self._nade.build_metrics(
                targets, predictions, cond_probs, log_probs
            )

        return metrics, metrics_upd, summaries

    def zero_state(self, batch_size):
        """Creates the initial RNN-NADE state, an RnnEstimatorStateTuple of zeros.

        Args:
            batch_size: batch size.

        Returns:
             An RnnEstimatorStateTuple of zeros.
        """
        with tf.variable_scope(f'{get_current_scope()}RnnRbmZeroState/', values=[batch_size]):
            initial_rnn_state = self._get_rnn_zero_state(batch_size)
            return RnnEstimatorStateTuple(
                tf.zeros((batch_size, self._nade.num_hidden), name='b_enc'),
                tf.zeros((batch_size, self.num_dims), name='b_dec'),
                initial_rnn_state
            )

    def _get_state(self,
                   inputs,
                   lengths=None,
                   initial_state=None,
                   last_outputs=False):
        """Computes the new state of the RNN-NADE.

        Scans through the inputs and returns the final state that either
        considers only the last outputs, or the outputs for all sequence steps.

        The final state consists of Estimator bias parameters and RNN cell's state.

        Args:
            inputs: A batch of sequences to process,
                sized `[batch_size, time_steps, num_input_dims]`
                or `[batch_size, num_input_dims]`.
            lengths: The lengths of input sequences sized `[batch_size]`,
                or None if all are equal.
            initial_state: An RnnEstimatorStateTuple, the initial state of the RNN-NADE,
                or None if the zero state should be used.
            last_outputs: `bool` indicating whether to return the outputs
                only from the last time step, or from all sequence steps.

        Returns:
            final_state: An RnnEstimatorStateTuple, the final state of the RNN-NADE.
        """
        # Processing the `lengths` and `initial_state` arguments
        lengths, initial_rnn_state = self._get_state_helper(
            inputs, lengths, initial_state
        )

        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=inputs,
            sequence_length=lengths)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=self._rnn_cell,
            helper=helper,
            initial_state=initial_rnn_state,
            output_layer=self._fc_layer)

        # Scanning through the input sequences
        with tf.variable_scope(get_current_scope()):
            final_outputs, final_rnn_state = tf.contrib.seq2seq.dynamic_decode(
                decoder
            )[0:2]

        # Final flattened outputs from the RNN-Dense decoder
        with tf.variable_scope('final_outputs'):
            if last_outputs:
                final_outputs_flat = final_outputs.rnn_output[:, -1, :]
            else:
                final_outputs_flat = flatten_maybe_padded_sequences(final_outputs.rnn_output, lengths)

        # Compute NADE biases from the Dense layer outputs.
        with tf.variable_scope('nade_biases'):
            b_enc, b_dec = self._build_biases(final_outputs_flat)

        with tf.variable_scope('final_state'):
            return RnnEstimatorStateTuple(b_enc, b_dec, final_rnn_state)

    def _build_biases(self, outputs):
        """Builds NADE biases from the Dense layer outputs.

        Args:
            outputs: Dense layer outputs,
                sized `[batch_size, num_dims + num_hidden[-1]]`.

        Returns:
            b_enc: Encoder bias terms, sized `[batch_size, num_hidden[-1]]`.
            b_dec: Decoder bias terms, sized `[batch_size, num_dims]`.
        """
        b_enc, b_dec = tf.split(outputs, [self._nade.num_hidden, self._nade.num_dims], axis=1)

        if self.internal_bias:
            b_enc = self._nade.b_enc + b_enc
            b_dec = self._nade.b_dec + b_dec

        return b_enc, b_dec

    def single_step(self, inputs, initial_state):
        """Processes the input sequences of one time step size.

        Args:
            inputs: A batch of single step sequences to process,
                sized `[batch_size, num_input_dims]`.
            initial_state: An RnnEstimatorStateTuple, the initial state of the RNN-NADE,
                or None if the zero state should be used.

        Returns:
            new_state: The new RNN-NADE state tuple after processing the inputs.
        """
        # Forward pass through the RNN cell
        rnn_outputs, rnn_state = self._rnn_cell(inputs, initial_state.rnn_state)

        # Forward pass through the Dense layer
        with tf.variable_scope('final_outputs'):
            outputs_flat = self._fc_layer(rnn_outputs)

        # Building NADE biases
        with tf.variable_scope('nade_biases'):
            b_enc, b_dec = self._build_biases(outputs_flat)

        with tf.variable_scope('final_state'):
            return RnnEstimatorStateTuple(b_enc, b_dec, rnn_state)

    def log_prob(self, inputs, targets_flat, lengths=None):
        """Computes the log probability of a sequence of values.

        Args:
            inputs: A batch of sequences to compute the log probabilities of,
                sized `[batch_size, time_steps, num_input_dims]`.
            targets_flat: The flattened target predictions,
                sized `[sum(lengths), num_dims]`
            lengths: The length of each sequence, sized `[batch_size]`
                or None if all are equal.

        Returns:
            log_prob: The log probability of each sequence value,
                sized `[sum(lengths), 1]`.
            cond_prob: The conditional probabilities for targets,
                sized `[sum(lengths), num_dims]`.
        """
        assert self.num_dims == targets_flat.shape[1].value

        with tf.name_scope(f'{get_current_scope()}{self.track_name}/'):
            state = self._get_state(inputs, lengths=lengths)

        with tf.variable_scope(f'log_probs/{self.track_name}'):
            return self._nade.log_prob(targets_flat, state.b_enc, state.b_dec)

    def sample_single(self, inputs, state):
        """Computes a sample and its probability from a batch of states.

        Args:
            inputs: (Unused) A batch of inputs from which sampling may start.
            state: An RnnEstimatorStateTuple containing the RNN-NADE state
                for each sample in the batch

        Returns:
            eval: A sample for each input state, sized `[batch_size, num_dims]`.
            log_prob: The log probability of each sample, sized `[batch_size, 1]`.
        """
        sample, log_prob = self._nade.sample(state.b_enc, state.b_dec, temperature=1.)

        return sample, log_prob

    def pretrain(self, optimizer, lr, run_optimizer=True):
        """Constructs training ops for pre-training the RNN-NADE.

        RNN-NADE does not require pre-training,
        so the function just returns empty update ops.
        """
        return [], [], self.metrics, self.metrics_upd, self.summaries
