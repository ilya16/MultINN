"""Implementation of a RNN-RBM generator."""

import tensorflow as tf

from models.common.rbm import RBM
from models.generators.rnn_estimator import RnnEstimator, RnnEstimatorStateTuple
from utils.auxiliary import get_current_scope
from utils.sequences import flatten_maybe_padded_sequences


class RnnRBM(RnnEstimator):
    """RNN-RBM generator [1].

    [1]: http://www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf
    """

    def __init__(self,
                 num_dims,
                 num_hidden,
                 num_hidden_rnn,
                 keep_prob=1.0,
                 internal_bias=True,
                 k=10,
                 name='rnn-rbm',
                 track_name='all'):
        """Initializes the RNN-RBM model.

        Args:
            num_dims: The number of output dimensions of the model.
            num_hidden: The number of hidden units of the RBM.
            num_hidden_rnn: The number of hidden units of the temporal RNN cell.
            keep_prob: The portion of outputs to use during the training,
                inverse of `dropout_rate`.
            internal_bias: `bool` indicating whether the Generator should maintain
                its own bias variables. If `False`, external values must be passed.
            k: The number of Gibbs sampling steps for the RBM.
            name: The model name to use as a prefix when adding operations.
            track_name: The track name on which encoder operates.
        """
        super().__init__(
            num_dims=num_dims, num_hidden=num_hidden, num_hidden_rnn=num_hidden_rnn,
            keep_prob=keep_prob, internal_bias=internal_bias,
            name=name, track_name=track_name
        )

        self._k = k

    def _init_estimator(self):
        """Initializes the RBM Estimator and the Dense layers that connects RNN and RBM."""
        initializer = tf.contrib.layers.xavier_initializer()

        self._rbm = RBM(num_dims=self.num_dims, num_hidden=self.num_hidden[-1], k=self.k)

        self._Wuh = tf.Variable(
            initializer([self.num_hidden_rnn[-1], self.num_hidden[-1]]),
            dtype=tf.float32,
            name="Wuh"
        )

        self._Wuv = tf.Variable(
            initializer([self.num_hidden_rnn[-1], self.num_dims]),
            dtype=tf.float32,
            name="Wuv"
        )

    @property
    def k(self):
        """The number of hidden units for all layers in the model."""
        return self._k

    def build(self, x=None, y=None, lengths=None, is_train=None, mode='eval'):
        """Builds RNN-RBM's graph.

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

            # Flatten inputs into the shape `[sum(lengths), num_dims]`
            with tf.variable_scope(f'{get_current_scope()}generator_inputs/{self.track_name}/flat/'):
                inputs_flat = flatten_maybe_padded_sequences(inputs, lengths)

            # Flatten targets into the shape `[sum(lengths), num_dims]`
            with tf.variable_scope(f'{get_current_scope()}generator_targets/{self.track_name}/flat/'):
                targets_flat = flatten_maybe_padded_sequences(targets, lengths)

            # Scanning through the input sequences.
            with tf.name_scope(f'{get_current_scope()}{self.name}/'):
                initial_state = self.zero_state(batch_size=tf.shape(x)[0])
                final_state = self._get_state(inputs, initial_state=initial_state)

            # Sampling outputs from the RBM
            with tf.variable_scope(f'{get_current_scope()}generator_outputs/{self.track_name}/'):
                self._outputs, cond_probs = self.sample_single(inputs_flat, final_state)

            # Building model metrics and summaries from the targets and outputs
            self._metrics, self._metrics_upd, self._summaries['metrics'] = self.build_metrics(
                targets=targets_flat,
                predictions=self._outputs,
                cond_probs=cond_probs,
            )

        # RNN-RBM variables
        self._variables = {
            **self._rbm.variables,
            'Wuh': self._Wuh, 'Wuv': self._Wuv,
            **self._rnn.variables,
        }

        self._trainable_variables = \
            self._rbm.trainable_variables \
            + self._rnn.trainable_variables \
            + [self._Wuh, self._Wuv]

        self._summaries['weights'] = self.weight_summary

        self._is_built = True

    def build_metrics(self, targets, predictions, cond_probs=None, log_probs=None):
        """Builds RNN-RBM metrics ops based on predictions and target values.

        Uses internal RBM to compute the metrics.

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
            metrics, metrics_upd, summaries = self._rbm.build_metrics(
                targets, predictions, cond_probs, log_probs
            )

        return metrics, metrics_upd, summaries

    def zero_state(self, batch_size):
        """Creates the initial RNN-RBM state, an RnnEstimatorStateTuple of zeros.

        Args:
            batch_size: batch size.

        Returns:
             An RnnEstimatorStateTuple of zeros.
        """
        with tf.variable_scope(f'{get_current_scope()}RnnRbmZeroState/', values=[batch_size]):
            initial_rnn_state = self._get_rnn_zero_state(batch_size)
            return RnnEstimatorStateTuple(
                tf.zeros((batch_size, self._rbm.num_hidden), name='bh'),
                tf.zeros((batch_size, self.num_dims), name='bv'),
                initial_rnn_state
            )

    def _get_state(self,
                   inputs,
                   lengths=None,
                   initial_state=None,
                   last_outputs=False):
        """Computes the new state of the RNN-RBM.

        Scans through the inputs and returns the final state that either
        considers only the last outputs, or the outputs for all sequence steps.

        The final state consists of Estimator bias parameters and RNN cell's state.

        Args:
            inputs: A batch of sequences to process,
                sized `[batch_size, time_steps, num_input_dims]`
                or `[batch_size, num_input_dims]`.
            lengths: The lengths of input sequences sized `[batch_size]`,
               or None if all are equal.
            initial_state: An RnnEstimatorStateTuple, the initial state of the RNN-RBM,
                or None if the zero state should be used.
            last_outputs: `bool` indicating whether to return the outputs
                only from the last time step, or from all sequence steps.

        Returns:
            final_state: An RnnEstimatorStateTuple, the final state of the RNN-RBM.
        """
        # Processing the `lengths` and `initial_state` arguments
        lengths, initial_rnn_state = self._get_state_helper(
            inputs, lengths, initial_state
        )

        # Scanning through the input sequences
        with tf.variable_scope(f'{get_current_scope()}{self.track_name}'):
            final_outputs, final_rnn_state = tf.nn.dynamic_rnn(
                self._rnn_cell, inputs,
                initial_state=initial_rnn_state,
                sequence_length=lengths,
                parallel_iterations=32,
                dtype=tf.float32
            )

        # Final flattened outputs from the RNN-Dense decoder
        with tf.variable_scope('final_outputs'):
            if last_outputs:
                final_outputs = final_outputs[:, -1, :]
                final_outputs_flat = tf.reshape(final_outputs, [-1, tf.shape(final_outputs)[-1]])
            else:
                final_outputs_flat = flatten_maybe_padded_sequences(final_outputs, lengths)

        # Compute RBM biases from the RNN cell outputs.
        with tf.variable_scope('rbm_biases'):
            bh_t, bv_t = self._build_biases(final_outputs_flat)

        with tf.variable_scope('final_state'):
            return RnnEstimatorStateTuple(bh_t, bv_t, final_rnn_state)

    def _build_biases(self, outputs):
        """Builds RNN biases from the RNN cell outputs.

        Args:
            outputs: RNN cell outputs,
                sized `[batch_size, num_hidden_rnn[-1]]`.

        Returns:
            bh: Hidden bias terms, sized `[batch_size, num_hidden[-1]]`.
            bv: Visible bias terms, sized `[batch_size, num_dims]`.
        """

        bh = tf.matmul(outputs, self._Wuh)
        bv = tf.matmul(outputs, self._Wuv)

        if self.internal_bias:
            bh = self._rbm.bh + bh
            bv = self._rbm.bv + bv

        return bh, bv

    def single_step(self, inputs, initial_state):
        """Processes the input sequences of one time step size.

        Args:
            inputs: A batch of single step sequences to process,
                sized `[batch_size, num_input_dims]`.
            initial_state: An RnnEstimatorStateTuple, the initial state of the RNN-RBM,
                or None if the zero state should be used.

        Returns:
            new_state: The new RNN-RBM state tuple after processing the inputs.
        """
        # Forward pass through the RNN cell
        rnn_outputs, rnn_state = self._rnn_cell(inputs, initial_state.rnn_state)

        # Building RBM biases
        with tf.variable_scope('rbm_biases'):
            bh_t, bv_t = self._build_biases(rnn_outputs)

        with tf.variable_scope('final_state'):
            return RnnEstimatorStateTuple(bh_t, bv_t, rnn_state)

    def sample_single(self, inputs, state):
        """Computes a sample and its probability from a batch of inputs and states.

        Args:
            inputs: Optional batch of inputs from which sampling in RBM start.
            state: An RnnEstimatorStateTuple containing the RNN-RBM state
                for each sample in the batch

        Returns:
            eval: A sample for each input state, sized `[batch_size, num_dims]`.
            cond_prob: The conditional probability for each sample, sized `[batch_size, num_dims]`.
        """
        cond_prob, sample = self._rbm.sample(inputs, state.b_enc, state.b_dec)

        return sample, cond_prob

    def pretrain(self, optimizer, lr, run_optimizer=True):
        """Constructs training ops for pre-training the Generator.

        RNN-RBM allows pre-training the RBM module.

        Args:
            optimizer: tf.Optimizer object that computes gradients.
            lr: A learning rate for weight updates.
            run_optimizer: `bool` indicating whether to run optimizer
                on the generator's trainable variables, or return
                an empty update ops list. Allows running the optimizer
                on the top of multiple generators.

        Returns:
            init_ops: A list of weight initialization ops.
            update_ops: A list of weights update ops.
            metrics: A dictionary of metric names and metric ops.
            metrics_upd: A list of metric update ops.
            summaries: A dict of tf.Summary objects for model's
                metrics, weights and training gradients.
        """
        inputs_flat = flatten_maybe_padded_sequences(self._inputs, self._lengths)

        return self._rbm.train(inputs_flat, lr)
