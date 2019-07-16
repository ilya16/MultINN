"""Implementation of a Feedback-RNN MultINN model with per-track modules and RNN feedback module."""

import tensorflow as tf

from models.common.rnn import RNN
from models.multinn.multinn_feedback import MultINNFeedback


class MultINNFeedbackRnn(MultINNFeedback):
    """Feedback-RNN MultINN model.

    Multiple per-track Encoders + Multiple per-track Generators
    + RNN Feedback module on the Generators layer.

    Works with multi-track sequences and learns the track specific features.
    The Feedback layer offers inter-track relation modelling.
    The Generators work with own track sequences and receive
    the processed feedback about the whole multi-track sequence.

    RNN Feedback module adds the history about the whole multi-track
    sequence compared to raw the Feedback MultINN model.
    """

    def __init__(self, config, params, name='MultINN-feedback-rnn'):
        """Initializes Feedback-RNN MultiNN model.

        Args:
            config: A dict of the experiment configurations.
            params: A dict of the model parameters.
            name: Optional model name to use as a prefix when adding operations.
        """
        super().__init__(config, params, name=name)

    def _init_feedback(self):
        """Initializes RNN Feedback module of the Feedback-RNN MultINN model."""
        self._feedback_layer = RNN(
            num_units=self._params['generator']['feedback'],
            keep_prob=self.keep_prob
        )

    def _apply_feedback(self, inputs, lengths=None, initial_state=None, single_step=False):
        """Runs the RNN Feedback module over the stacked encoded inputs.

        Args:
            inputs: A batch of stacked encoded inputs,
                sized `[batch_size, time_steps, num_tracks x num_dims]`,
                or    `[batch_size, num_tracks x num_dims]`.
            lengths: The lengths of input sequences sized `[batch_size]`,
                or None if all are equal.
            initial_state: The initial state of the Feedback module,
                or None if the zero state should be used
                or there is no Feedback module state.
            single_step: `bool` indicating whether to run the Feedback module
                over a single step sequence or not.

        Returns:
             feedback_outputs: A batch of feedback vectors,
                sized [batch_size, time_steps, num_feedback].
        """
        if lengths is None:
            lengths = tf.tile(tf.shape(inputs)[1:2], [tf.shape(inputs)[0]])

        if initial_state is None:
            initial_state = self._feedback_layer.zero_state(batch_size=tf.shape(inputs)[0])

        rnn_cell = self._feedback_layer.cell

        if single_step:
            final_outputs, final_state = rnn_cell(inputs, initial_state)
        else:
            final_outputs, final_state = tf.nn.dynamic_rnn(
                rnn_cell, inputs,
                initial_state=initial_state,
                sequence_length=lengths,
                parallel_iterations=32,
                dtype=tf.float32
            )

        return final_outputs, final_state
