"""Implementation of a RNN (Recurrent Neural Network).

TensorFlow RNN cell is wrapped into a Python class to
reuse the proposed interfaces for models.
"""

import tensorflow as tf

from models.common.model import Model
from utils.auxiliary import get_current_scope


class RNN(Model):
    """Recurrent Neural Network.

    Multi-layer RNN cell.
    """

    def __init__(self,
                 num_units=128,
                 keep_prob=1.,
                 attn_length=0,
                 learn_zero_state: bool = False,
                 name='rnn'):
        """Initializes RNN.

        Args:
            num_units: The number of hidden units of the RNN,
                or a list of hidden units for multiple layers.
            keep_prob: The portion of outputs to use during the training,
                inverse of `dropout_rate`.
            attn_length: The size of the attention window.
                no attention wrapper is added if value is zero.
            learn_zero_state: `bool` indicating whether to learn the zero state
                of the RNN cell or use zero vectors.
            name: The name for the RNN model.
        """
        super().__init__(name=name)

        if isinstance(num_units, int):
            num_units = [num_units]

        self._num_units = num_units

        self._learn_zero_state = learn_zero_state
        self._keep_prob = keep_prob
        self._attn_length = attn_length

        # RNN cell and initial states
        self.cell = None
        self.c0, self.h0 = None, None

        self._is_built = False

    @property
    def num_units(self):
        """list of int: The number of hidden units for all layers of the RNN."""
        return self._num_units

    @property
    def num_layers(self):
        """int: The number of layers in the RNN."""
        return len(self._num_units)

    @property
    def learn_zero_state(self):
        """bool: Whether the zero state is learned or not."""
        return self._learn_zero_state

    @property
    def keep_prob(self):
        """float: Dropout keep probability."""
        return self._keep_prob

    @property
    def attn_length(self):
        """int: The size of the attention window."""
        return self._attn_length

    def build(self, x=None, y=None, lengths=None, is_train=None, mode='eval'):
        """Builds RNN.

        Since RNN cell variables are initialized only on the first call,
        the `build()` function only defines the cell objects and operates
        with `is_train` variable that conditions the `DropoutWrapper`.

        Args:
            x: (Unused) inputs to the model,
                sized `[batch_size, time_steps, num_input_dims]`.
            y: (Unused) targets for the model,
                sized `[batch_size, time_steps, num_output_dims]`.
            lengths: (Unused) lengths of input sequences sized `[batch_size]`.
            is_train: `bool` for distinguishing between training and inference.
            mode: Build mode for optimizing the graph size.
                Some operations may not be needed for training or inference.

         Raises:
            ValueError: If incorrect build mode is provided.
        """
        super().build(x, y, lengths, is_train, mode)

        self.build_cell(is_train)

    def build_cell(self, is_train):
        """Builds Multi-layer RNN cell.

        Can be called externally and is called on the `build()` function call.

        Args:
            is_train: `bool` for distinguishing between training and inference.
        """
        if not self.is_built:
            with tf.variable_scope(f'{get_current_scope()}{self.name}/'):
                cells = []

                # Dropout output probability.
                if is_train is None:
                    output_keep_prob = self.keep_prob
                else:
                    output_keep_prob = tf.cond(is_train, lambda: self.keep_prob, lambda: 1.0)

                # Creating `self.num_layers` RNN cells.
                for i in range(self.num_layers):
                    cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.num_units[i])

                    # Adding attention wrapper.
                    if self._attn_length > 0 and not cells:
                        cell = tf.contrib.rnn.AttentionCellWrapper(
                            cell, self._attn_length, state_is_tuple=True)

                    # Adding dropout wrapper.
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=output_keep_prob)

                    cells.append(cell)

                # Creating Multi-layer RNN cell.
                self.cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

                # Creating variables for zero state if it is learned by the model.
                if self._learn_zero_state:
                    self.c0 = [tf.Variable(tf.zeros([1, self._num_units[i]], tf.float32), name=f"cell_{i}/c0")
                               for i in range(self.num_layers)]
                    self.h0 = [tf.tanh(self.c0[i], name=f"cell_{i}/h0") for i in range(self.num_layers)]

            self._is_built = True

    def build_metrics(self, targets, predictions, cond_probs=None, log_probs=None):
        """Building RNN metrics ops based on predictions and target values.

        Note:
            RNN does not define any metrics on the output layer.
        """
        return [], [], None

    def zero_state(self, batch_size, dtype=tf.float32):
        """Returns the initial state of the RNN cell.

        Args:
            batch_size: The batch size of inputs to the cell.
            dtype: The type of the initial state.

        Returns:
            initial_state: RNN cell's initial state.
        """
        with tf.variable_scope(f'{get_current_scope()}{self.name}/'):
            if self._learn_zero_state:
                initial_state = tuple(
                    tf.contrib.rnn.LSTMStateTuple(
                        tf.tile(self.c0[i], [batch_size, 1], name=f"cell_{i}/c0_batch"),
                        tf.tile(self.h0[i], [batch_size, 1], name=f"cell_{i}/h0_batch"),
                    ) for i in range(self.num_layers)
                )
            else:
                initial_state = self.cell.get_initial_state(batch_size=batch_size, dtype=dtype)

        return initial_state

    def __call__(self, inputs, state, *args, **kwargs):
        """Calls RNN cell

        Args:
            inputs: A batch of inputs to the RNN cell,
                sized `[batch_size, num_dims]`.
            state: RNN cell state tuple.

        Returns:
            outputs: The outputs from the RNN cell,
                sized `[batch_size, num_hidden[-1]]`.
            new_state: A new RNN cell state tuple.
        """
        if not self._is_built:
            raise RuntimeError('RNN cell is not built yet, build it with `cell.build_cell()` before calling')

        return self.cell(inputs, state)

    @property
    def variables(self):
        """A dictionary of RNN's variables grouped by modules.

        Since RNN cell variables are created only on the first cell call,
        the variables are not initialized inside `build_cell()` function
        and the dictionary is created on each new property access.
        """
        return {'rnn': self.cell.trainable_variables, 'c0': self.c0}

    @property
    def trainable_variables(self):
        """A list of RNN's trainable variables.

        Since RNN cell variables are created only on the first cell call,
        the variables are not initialized inside `build_cell()` function
        and the dictionary is created on each new property access.
        """
        self._trainable_variables = self.cell.trainable_variables

        if self._learn_zero_state:
            self._trainable_variables += self.c0

        return self._trainable_variables
