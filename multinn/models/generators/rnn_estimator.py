"""Implementation of an interface for RNN-Estimator models."""

import abc
import collections

import tensorflow as tf

from models.common.rnn import RNN
from models.generators.generator import Generator
from utils.auxiliary import get_current_scope

_RnnEstimatorStateTuple = collections.namedtuple(
    'RnnEstimatorStateTuple', ('b_enc', 'b_dec', 'rnn_state')
)


class RnnEstimatorStateTuple(_RnnEstimatorStateTuple):
    """Tuple used by Rnn-Estimator models to store state.

    Stores three elements `(b_enc, b_dec, rnn_state)`, in that order:
        b_enc: Estimator encoder (hidden) bias terms,
            sized `[batch_size, num_hidden]`.
        b_dec: Estimator decoder (visible) bias terms,
            sized `[batch_size, num_dims]`.
        rnn_state: The RNN cell's state.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (b_enc, b_dec, rnn_state) = self
        if not b_enc.dtype == b_dec.dtype == rnn_state.dtype:
            raise TypeError(
                'Inconsistent internal state: %s vs %s vs %s' %
                (str(b_enc.dtype), str(b_dec.dtype), str(rnn_state.dtype)))
        return b_enc.dtype


class RnnEstimator(Generator):
    """RNN-Estimator models' [1] interface.

    Example models:
        RNN-RBM [1]
        RNN-NADE [1]

    The models consist of a temporal recurrent unit for modelling sequence
    dependencies and a distribution estimator for modelling multi-modal
    distributions of inputs.

    Note:
        The model differs from original RNN-RBM models by allowing
        having the inputs to RNN and Estimator of different shapes.
        It allows adding additional labels to sequences while modelling
        the multi-modal distributions of target values.

    [1]: http://www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf
    """

    def __init__(self,
                 num_dims,
                 num_hidden,
                 num_hidden_rnn,
                 keep_prob=1.0,
                 internal_bias=True,
                 name='rnn-rbm',
                 track_name='all'):
        """Initializes the base parameters and variables of the RNN-RBM models.

        Args:
            num_dims: The number of output dimensions of the model.
            num_hidden: The number of hidden units of the Estimator.
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

        with tf.name_scope(f'{get_current_scope()}{self.name}/{self.track_name}/'):
            self._rnn_cell = None
            self._init_rnn()

            self._init_estimator()

    def _init_rnn(self):
        """Initializes the RNN cell"""
        self._rnn = RNN(num_units=self.num_hidden_rnn, keep_prob=self.keep_prob)

    @abc.abstractmethod
    def _init_estimator(self):
        """Initializes the Estimator model.

        Each model implementation should define the function explicitly.
        """
        pass

    @abc.abstractmethod
    def build(self, x=None, y=None, lengths=None, is_train=None, mode='eval'):
        """Abstract function that builds the RNN-RBM model.

        Each Rnn-Estimator implementation should define the function explicitly.

        The function should assign the model's variables, trainable variables,
        placeholders, metrics, and summaries unless these attributes are assigned
        during the model initialization or these attributes are not needed by
        the encoder's design.

        Each model should assign `self._is_built = True` to assure that
        generator is built and can be used for training or inference.

        The core implementation only builds the common RNN cell.

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

        with tf.name_scope(f'{get_current_scope()}{self.name}/{self.track_name}/'):
            self._rnn.build_cell(is_train=is_train)
            self._rnn_cell = self._rnn.cell

    @abc.abstractmethod
    def zero_state(self, batch_size):
        """Creates the initial model's state, an RnnEstimatorStateTuple of zeros.

        Each RnnEstimator implementation should define the function explicitly.

        Args:
            batch_size: batch size.

        Returns:
             An RnnEstimatorStateTuple of zeros.
        """
        pass

    def _get_rnn_zero_state(self, batch_size):
        """Returns RNN cell's initial state"""
        return self._rnn.zero_state(batch_size, tf.float32)

    @abc.abstractmethod
    def _get_state(self,
                   inputs,
                   lengths=None,
                   initial_state=None,
                   last_outputs=False):
        """Abstract function for computing the new state of the RNN-Estimator.

        Scans through the inputs and returns the final state that either
        considers only the last outputs, or the outputs for all sequence steps.

        The final state consists of Estimator bias parameters and RNN cell's state.

        Each RnnEstimator implementation should define the function explicitly.

        Args:
            inputs: A batch of sequences to process,
                sized `[batch_size, time_steps, num_input_dims]`
                or `[batch_size, num_input_dims]`.
            lengths: The lengths of input sequences sized `[batch_size]`,
                or None if all are equal.
            initial_state: An RnnEstimatorStateTuple, the initial state of the RNN-Estimator,
                or None if the zero state should be used.
            last_outputs: `bool` indicating whether to return the outputs
                only from the last time step, or from all sequence steps.

        Returns:
            final_state: An RnnEstimatorStateTuple, the final state of the RNN-Estimator.
        """
        pass

    def _get_state_helper(self, inputs, lengths=None, initial_state=None):
        """Processes the `lengths` and `initial_state` for the `_get_state().`

        Args:
            inputs: A batch of sequences to compute the state from,
                sized `[batch_size, time_steps, num_input_dims]`
                or `[batch_size, num_input_dims]`.
            lengths: The lengths of input sequences sized `[batch_size]`,
                or None if all are equal.
            initial_state: An RnnEstimatorStateTuple, the initial state of the RNN-Estimator,
                or None if the zero state should be used.

        Returns:
            lengths: The lengths of input sequences sized `[batch_size]`.
            initial_state: An RnnEstimatorStateTuple, the initial state of the RNN-Estimator.
        """
        with tf.variable_scope('batch_size'):
            batch_size = tf.shape(inputs)[0]

        with tf.variable_scope('lengths'):
            if lengths is None:
                lengths = tf.tile(tf.shape(inputs)[1:2], [batch_size])

        with tf.variable_scope('initial_state'):
            if initial_state is None:
                initial_rnn_state = self._get_rnn_zero_state(batch_size)
            else:
                initial_rnn_state = initial_state.rnn_state

        return lengths, initial_rnn_state

    def steps(self, inputs, initial_state=None):
        """Scans through the input sequences and returns the final state.

        Args:
            inputs: A batch of sequences to process,
                sized `[batch_size, time_steps, num_input_dims]`
                or `[batch_size, num_input_dims]`.
            initial_state: An RnnEstimatorStateTuple, the initial state of the RNN-Estimator,
                or None if the zero state should be used.

        Returns:
            new_state: The new RNN-Estimator state tuple after processing the inputs.
        """
        return self._get_state(inputs, initial_state=initial_state, last_outputs=True)

    @abc.abstractmethod
    def single_step(self, inputs, initial_state):
        """Processes the input sequences of one time step size.

        Each RnnEstimator implementation should define the function explicitly.

        Args:
            inputs: A batch of single step sequences to process,
                sized `[batch_size, num_input_dims]`.
            initial_state: An RnnEstimatorStateTuple, the initial state of the RNN-Estimator,
                or None if the zero state should be used.

        Returns:
            new_state: The new RNN-Estimator state tuple after processing the inputs.
        """
        pass

    @abc.abstractmethod
    def sample_single(self, inputs, state):
        """Computes a sample and its probability from a batch of inputs and states.

        Each RnnEstimator implementation should define the function explicitly.

        Args:
            inputs: Optional batch of inputs from which sampling may start.
            state: An RnnEstimatorStateTuple containing the RNN-Estimator state
                for each sample in the batch

        Returns:
            eval: A sample for each input state, sized `[batch_size, num_dims]`.
            log_prob: The log probability of each sample, sized `[batch_size, 1]`.
        """
        pass

    def generate(self, x, num_steps):
        """Generates new sequences.

        The generative process starts by going through the batch of
        input sequences `x` and obtaining the final inputs' state.
        Then, the generator generates `num_steps` new steps by sampling
        from the modelled conditional distributions.

        Args:
            x: A batch of inputs to the generator,
                sized `[batch_size, time_steps, ...]`.
            num_steps: A number of time steps to generate.

        Returns:
            samples: A batch of generated sequences,
                sized `[batch_size, num_steps, num_output_dims]`
        """
        intro_state = self._get_state(x, lengths=None, last_outputs=True)

        intro = tf.reshape(x[:, -1, :], [tf.shape(x)[0], tf.shape(x)[-1]])

        # Sampling recursively `num_steps` long sequences
        samples, _ = tf.scan(
            self._generate_recurrence, tf.zeros((num_steps, 1)),
            initializer=(intro, intro_state)
        )

        return tf.transpose(samples, [1, 0, 2])

    def _generate_recurrence(self, val, _):
        """Auxiliary function for recursive sampling.

        Samples from the passed state and computes the next RNN-Estimator state.

        Args:
            val: A tuple (intro, state) of samples from the last time step
                and the current RnnEstimatorStateTuple.

        Returns:
            samples: The samples for the current state,
                sized `[batch_size, num_output_dims]`
            state: The new RnnEstimatorStateTuple.
        """
        # Unpacking previous step data
        intro, state = val

        # Sampling values
        samples, _ = self.sample_single(intro, state)

        # Computing new state
        state = self.single_step(samples, state)

        return samples, state
