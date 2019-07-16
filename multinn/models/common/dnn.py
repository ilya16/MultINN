"""Implementation of a DNN (Dense Neural Network).

TensorFlow Dense layer is wrapped into a Python class to
reuse the proposed interfaces for models.
"""

import itertools

import tensorflow as tf

from models.common.model import Model
from utils.auxiliary import get_current_scope


class DNN(Model):
    """Dense Neural Network.

    Multi-layer Fully-Connected Neural Network.
    """

    def __init__(self,
                 num_units=128,
                 activation=tf.nn.sigmoid,
                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                 name='dnn'):
        """Initializes DNN.

        Args:
            num_units: The number of hidden units of the DNN,
                or a list of hidden units for multiple layers.
            activation: The activation function for each layer.
            kernel_initializer: The weight initializer.
            name: The name for the RNN model.
        """
        super().__init__(name=name)

        if isinstance(num_units, int):
            num_units = [num_units]

        self._num_units = num_units

        self._activation = activation
        self._kernel_initializer = kernel_initializer

        self._fc_layers = []
        self._init_layers()

        self._is_built = False

    def _init_layers(self):
        """Initializes the layers of the DNN."""
        with tf.variable_scope(f'{get_current_scope()}{self.name}/'):
            self._fc_layers = []

            for i in range(self.num_layers):
                layer = tf.layers.Dense(
                    units=self.num_units[i],
                    activation=self._activation,
                    kernel_initializer=self._kernel_initializer
                )

                self._fc_layers.append(layer)

    @property
    def num_units(self):
        """list of int: The number of hidden units for all layers of the DNN."""
        return self._num_units

    @property
    def num_layers(self):
        """int: The number of layers in the DNN."""
        return len(self._num_units)

    def build(self, x=None, y=None, lengths=None, is_train=None, mode='eval'):
        """Builds DNN's Graph.

        Since all variables of DNN are defined during the model initialization
        and DNN is a base model mostly used by other models that define
        the operation graph, the `build()` function only updates the conventional
        variable `is_built` and is called inside `__init__()`.

        Raises:
            ValueError: If incorrect build mode is provided.
        """
        super().build(x, y, lengths, is_train, mode)

        self._is_built = True

    def build_metrics(self, targets, predictions, cond_probs=None, log_probs=None):
        """Building DNN metrics ops based on predictions and target values.

        Note:
            DNN does not define any metrics on the output layer.
        """
        pass

    def __call__(self, inputs, *args, **kwargs):
        """Forward pass in DNN.

        Args:
            inputs: A batch of inputs to the DNN,
                sized `[batch_size, num_dims]`.

        Returns:
            outputs: The outputs from the DNN,
                sized `[batch_size, num_hidden[-1]]`.
        """
        if not self._is_built:
            raise RuntimeError('DNN is not built yet, '
                               'build it using `dnn.build()` before calling')

        x = inputs
        for i in range(self.num_layers):
            x = self._fc_layers[i](x)

        return x

    @property
    def variables(self):
        """A dictionary of DNN's variables grouped by modules.

        Since DNN's variables are created only on the first network call,
        the variables are not initialized inside `build()` function
        and the dictionary is created on each new property access.
        """
        return {'dense': self.trainable_variables}

    @property
    def trainable_variables(self):
        """A list of DNN's trainable variables.

        Since DNN's variables are created only on the first network call,
        the variables are not initialized inside `build()` function
        and the dictionary is created on each new property access.
        """
        return list(itertools.chain.from_iterable(
            [layer.trainable_variables for layer in self._fc_layers]
        ))
