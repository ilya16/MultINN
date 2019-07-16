"""Implementation of a DBN (Deep Belief Network)."""

import itertools

import tensorflow as tf

from models.common.model import Model
from models.common.rbm import RBM
from utils.auxiliary import get_current_scope


class DBN(Model):
    """Deep Belief Network [1].

    A stacked network of multiple RBMs.

    [1]: https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf
    """

    def __init__(self,
                 num_dims,
                 num_hidden,
                 k=10,
                 name='dbn'):
        """Initializes DBN.

        Args:
            num_dims: The number of input/output dimensions of the DBN.
            num_hidden: The number of hidden units of the DBN (=RBM),
                or a list of hidden layer units.
            k: The number of steps to use in Gibbs sampling.
            name: The name for the DBN model.
        """
        super().__init__(name=name)

        self._num_dims = num_dims
        if isinstance(num_hidden, int):
            num_hidden = [num_hidden]
        self._num_hidden = num_hidden

        self._k = k
        self._rbm_layers = []

        with tf.name_scope(f'{get_current_scope()}{self.name}/'):
            # Initializing RBM layers
            for i in range(self.num_layers):
                input_dims = self.num_dims if i == 0 else self.num_hidden[i - 1]

                rbm = RBM(
                    num_dims=input_dims,
                    num_hidden=self.num_hidden[i],
                    k=self.k,
                    name=f'rbm/{i}'
                )
                self._rbm_layers.append(rbm)

        # DBN variables
        self._variables = [self.rbm_layers[i].variables for i in range(self.num_layers)]
        self._trainable_variables = [self.rbm_layers[i].trainable_variables for i in range(self.num_layers)]
        self._trainable_variables = list(itertools.chain.from_iterable(self._trainable_variables))

        self.build()

    @property
    def num_dims(self):
        """int: The number of input/output dimensions of the DBN."""
        return self._num_dims

    @property
    def num_hidden(self):
        """list of int: The number of hidden units for all layers of the DBN."""
        return self._num_hidden

    @property
    def k(self):
        """The number of Gibbs sampling steps (the length of Gibbs chain)."""
        return self._k

    @property
    def num_layers(self):
        """int: The number of layers in the DBN."""
        return len(self._num_hidden)

    @property
    def rbm_layers(self):
        """A list of RBMs"""
        return self._rbm_layers

    def build(self, x=None, y=None, lengths=None, is_train=None, mode='eval'):
        """Builds DBN's Graph.

        Since all variables of DBN are defined during the model initialization
        and DBN is a base model mostly used by other models that define
        the operation graph, the `build()` function only updates the conventional
        variable `is_built` and is called inside `__init__()`.

        Raises:
            ValueError: If incorrect build mode is provided.
        """
        super().build(x, y, lengths, is_train, mode)

        self._is_built = True

    def build_metrics(self, targets, predictions, cond_probs=None, log_probs=None, layer=None):
        """Builds RBM metrics ops based on predictions and target values.

        Args:
            targets: Flattened target predictions for the i-th DBN layer,
                sized `[batch_size, num_dims_i]`.
            predictions: Flattened model's predictions for the i-th DBN layer,
                sized `[batch_size, num_dims_i]`.
            cond_probs: Flattened conditional probabilities for predictions.
                sized `[batch_size, num_dims_i]`.
            log_probs: Flattened log probabilities for predictions,
                sized `[batch_size]`.
            layer: A layer index for which metrics should be computed,
                or None if metrics are computed on the global DBN level
                (first layer).

        Returns:
            metrics: A dictionary of metric names and metric ops.
            metrics_upd: A list of metric update ops.
            metric_summaries: A tf.Summary object for model's metrics.

        Raises:
            ValueError: If none of cond_probs and log_probs is provided.
        """
        layer = layer if layer is not None else 0

        metrics, metrics_upd, summaries = self.rbm_layers[layer].build_metrics(
            targets, predictions, cond_probs=cond_probs
        )

        return metrics, metrics_upd, summaries

    def forward(self, v, bh=None):
        """Forward pass of inputs in DBN.

        Args:
            v: A batch of inputs to RBM, sized `[batch_size, num_dims]`.
            bh: A list of external hidden bias terms, sized `[num_layers, batch_size, num_hidden]`,
                or None if the internal bias term should be used.

        Returns:
            p_h: The activation conditional probabilities on the hidden layer,
                sized `[batch_size, num_hidden]`.
            h: The sampled hidden layer activations,
                sized `[batch_size, num_hidden]`.
        """
        p_h, h = 0, v

        for i in range(len(self.rbm_layers)):
            bh_i = bh[i] if bh is not None else None
            p_h, h = self.rbm_layers[i].forward(h, bh=bh_i)

        return p_h, h

    def reconstruct(self, h, bv=None):
        """Backward pass of hidden vectors in DBN.

        Computes reconstruction of inputs from hidden values of the last layer.

        Args:
             h: A batch of hidden vector inputs, sized `[batch_size, num_hidden[-1]]`.
             bv: A list of external hidden bias terms, sized `[num_layers, batch_size, num_dims]`,
                or None if the internal bias term should be used.

        Returns:
            p_v: The activation conditional probabilities on the visible layer,
                sized `[batch_size, num_dims]`.
            v: The sampled visible layer activations,
                sized `[batch_size, num_dims]`.
        """
        p_v, v = 0, h

        for i in range(len(self.rbm_layers) - 1, -1, -1):
            bv_i = bv[i] if bv is not None else None
            p_v, v = self.rbm_layers[i].reconstruct(v, bv=bv_i)

        return p_v, v
