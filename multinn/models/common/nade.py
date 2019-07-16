"""Implementation of a NADE (Neural Autoreressive Distribution Estimator).

Note:
    Adapted from https://github.com/tensorflow/magenta/blob/master/magenta/common/nade.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import tensorflow_probability as tfp

from metrics.statistical import base_metrics, build_summaries
from models.common.model import Model
from utils.auxiliary import safe_log


class NADE(Model):
    """Neural Autoregressive Distribution Estimator [1].

    [1]: https://arxiv.org/abs/1605.02226
    """

    def __init__(self,
                 num_dims,
                 num_hidden=128,
                 internal_bias=False,
                 name='nade'):
        """
        Initializes NADE.

        Args:
            num_dims: The number of input/output dimensions for each observation.
            num_hidden: The number of hidden units of the NADE.
            internal_bias: `bool` indicating whether the model should maintain
                its own bias variables. If `False`, external values must be passed
                to `log_prob()` and `sample()`.
            name: The name for NADE model.
        """
        super().__init__(name=name)

        self._num_dims = num_dims
        self._num_hidden = num_hidden
        self._internal_bias = internal_bias

        std = 1.0 / math.sqrt(self._num_dims)
        initializer = tf.truncated_normal_initializer(stddev=std)

        with tf.variable_scope(self.name):
            # Encoder weights (`V` in [1]).
            self.w_enc = tf.Variable(
                initializer([self.num_dims, 1, self.num_hidden]),
                dtype=tf.float32,
                name='w_enc'
            )

            # Transposed decoder weights (`W'` in [1]).
            self.w_dec = tf.Variable(
                initializer([self.num_dims, self.num_hidden, 1]),
                dtype=tf.float32,
                name='w_dec'
            )

            # Internal encoder bias term (`c` in [1]).
            # If `internal_bias` is `True`, the value is added to the external bias.
            if internal_bias:
                self.b_enc = tf.Variable(
                    initializer([1, self.num_hidden]),
                    dtype=tf.float32,
                    name='b_enc'
                )
            else:
                self.b_enc = None

            # Internal decoder bias term (`b` in [1]).
            # If `internal_bias` is `True`, the value is added to the external bias.
            if internal_bias:
                self.b_dec = tf.Variable(
                    initializer([1, self.num_dims]),
                    dtype=tf.float32,
                    name='b_dec'
                )
            else:
                self.b_dec = None

        # NADE variables
        self._variables = {'w_enc': self.w_enc, 'w_dec': self.w_dec,
                           'b_enc': self.b_enc, 'b_dec': self.b_dec}

        self._trainable_variables = [self.w_enc, self.w_dec]
        if self.internal_bias:
            self._trainable_variables += [self.b_enc, self.b_dec]

        self.build()

    @property
    def num_dims(self):
        """int: The number of input/output dimensions of the NADE."""
        return self._num_dims

    @property
    def num_hidden(self):
        """int: The number of hidden units of the NADE."""
        return self._num_hidden

    @property
    def internal_bias(self):
        """bool: The number of hidden units for each input/output of the NADE."""
        return self._internal_bias

    def build(self, x=None, y=None, lengths=None, is_train=None, mode='eval'):
        """Builds NADE's Graph.

        Since all variables of NADE are defined during the model initialization
        and NADE is a base model mostly used by other models that define
        the operation graph, the `build()` function only updates the conventional
        variable `is_built` and is called inside `__init__()`.

        Raises:
            ValueError: If incorrect build mode is provided.
        """
        super().build(x, y, lengths, is_train, mode)

        self._is_built = True

    def build_metrics(self, targets, predictions, cond_probs=None, log_probs=None):
        """Builds NADE metrics ops based on predictions and target values.

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
        """
        # Using base model metrics.
        metrics, metrics_upd = base_metrics(log_probs, targets, predictions, log_probs)

        # Building tf.Summary object from metrics.
        summaries = build_summaries(metrics)

        return metrics, metrics_upd, summaries

    def log_prob(self, x, b_enc=None, b_dec=None):
        """Gets the log probability and conditionals for observations.

        Args:
            x: A batch of observations to compute the log probability of,
                sized `[batch_size, num_dims]`.
            b_enc: External encoder bias terms (`c` in [1]),
                sized `[batch_size, num_hidden]`,
                or None if the internal bias term should be used.
            b_dec: External decoder bias terms (`b` in [1]),
                sized `[batch_size, num_dims]`,
                or None if the internal bias term should be used.

        Returns:
            log_prob: The log probabilities of each observation in the batch,
                sized `[batch_size]`.
            cond_prob: The conditional probabilities at each index for every batch,
                sized `[batch_size, num_dims]`.

        Raises:
            ValueError: If none of b_enc and b_dec is provided and `internal_bias`
                is set `False`.
        """
        batch_size = tf.shape(x)[0]

        if (b_enc is None or b_dec is None) and not self.internal_bias:
            raise ValueError('Bias values should be provided when `internal_bias` is `False`')

        # Broadcast if needed.
        if b_enc.shape[0] == 1 != batch_size:
            b_enc = tf.tile(b_enc, [batch_size, 1])
        if b_dec.shape[0] == 1 != batch_size:
            b_dec = tf.tile(b_dec, [batch_size, 1])

        # Initial condition before the loop.
        a_0 = b_enc
        log_p_0 = tf.zeros([batch_size, 1])
        cond_p_0 = []

        x_arr = tf.unstack(tf.reshape(tf.transpose(x), [self.num_dims, batch_size, 1]))
        w_enc_arr = tf.unstack(self.w_enc)
        w_dec_arr = tf.unstack(self.w_dec)
        b_dec_arr = tf.unstack(tf.reshape(tf.transpose(b_dec), [self.num_dims, batch_size, 1]))

        def loop_body(i, a, log_p, cond_p):
            """Accumulate hidden state, log_p, and cond_p for index i."""
            # Get variables for time step.
            w_enc_i = w_enc_arr[i]
            w_dec_i = w_dec_arr[i]
            b_dec_i = b_dec_arr[i]
            v_i = x_arr[i]

            cond_p_i, _ = self._cond_prob(a, w_dec_i, b_dec_i)

            # Get log probability for this value. Log space avoids numerical issues.
            log_p_i = v_i * safe_log(cond_p_i) + (1 - v_i) * safe_log(1 - cond_p_i)

            # Accumulate log probability.
            log_p_new = log_p + log_p_i

            # Save conditional probabilities.
            cond_p_new = cond_p + [cond_p_i]

            # Encode value and add to hidden units.
            a_new = a + tf.matmul(v_i, w_enc_i)

            return a_new, log_p_new, cond_p_new

        # Build the actual loop
        a, log_p, cond_p = a_0, log_p_0, cond_p_0
        for i in range(self.num_dims):
            a, log_p, cond_p = loop_body(i, a, log_p, cond_p)

        return (-tf.squeeze(log_p, axis=[1]),
                tf.transpose(tf.squeeze(tf.stack(cond_p), [2])))

    def sample(self, b_enc=None, b_dec=None, n=None, temperature=None):
        """Generate samples for the batch from the NADE.

        Args:
            b_enc: External encoder bias terms (`c` in [1]),
                sized `[batch_size, num_hidden]`,
                or None if the internal bias term should be used.
            b_dec: External decoder bias terms (`b` in [1]),
                sized `[batch_size, num_dims]`,
                or None if the internal bias term should be used.
            n: The number of samples to generate,
                or None, if the batch size of `b_enc` should be used.
            temperature: The amount to divide the logits by before sampling
                each Bernoulli, or None if a threshold of 0.5 should be used
                instead of sampling.

        Returns:
            eval: The generated samples, sized `[batch_size, num_dims]`.
            log_prob: The log probabilities of each observation in the batch,
                sized `[batch_size]`.
        """
        if (b_enc is None or b_dec is None) and not self.internal_bias:
            raise ValueError('Bias values should be provided when `internal_bias` is `False`')

        batch_size = n or tf.shape(b_enc)[0]

        # Broadcast if needed.
        if b_enc.shape[0] == 1 != batch_size:
            b_enc = tf.tile(b_enc, [batch_size, 1])
        if b_dec.shape[0] == 1 != batch_size:
            b_dec = tf.tile(b_dec, [batch_size, 1])

        a_0 = b_enc
        sample_0 = []
        log_p_0 = tf.zeros([batch_size, 1])

        w_enc_arr = tf.unstack(self.w_enc)
        w_dec_arr = tf.unstack(self.w_dec)
        b_dec_arr = tf.unstack(tf.reshape(tf.transpose(b_dec), [self.num_dims, batch_size, 1]))

        def loop_body(i, a, sample, log_p):
            """Accumulate hidden state, sample, and log probability for index i."""
            # Get weights and bias for time step.
            w_enc_i = w_enc_arr[i]
            w_dec_i = w_dec_arr[i]
            b_dec_i = b_dec_arr[i]

            cond_p_i, cond_l_i = self._cond_prob(a, w_dec_i, b_dec_i)

            if temperature is None:
                v_i = tf.to_float(tf.greater_equal(cond_p_i, 0.5))
            else:
                bernoulli = tfp.distributions.Bernoulli(
                    logits=cond_l_i / temperature,
                    dtype=tf.float32
                )
                v_i = bernoulli.sample()

            # Accumulate sampled values.
            sample_new = sample + [v_i]

            # Get log probability for this value. Log space avoids numerical issues.
            log_p_i = v_i * safe_log(cond_p_i) + (1 - v_i) * safe_log(1 - cond_p_i)

            # Accumulate log probability.
            log_p_new = log_p + log_p_i

            # Encode value and add to hidden units.
            a_new = a + tf.matmul(v_i, w_enc_i)

            return a_new, sample_new, log_p_new

        a, sample, log_p = a_0, sample_0, log_p_0
        for i in range(self.num_dims):
            a, sample, log_p = loop_body(i, a, sample, log_p)

        return (tf.transpose(tf.squeeze(tf.stack(sample), [2])),
                -tf.squeeze(log_p, axis=[1]))

    def _cond_prob(self, a, w_dec_i, b_dec_i):
        """Gets the conditional probability for a single dimension.

        Args:
            a: Model's hidden state, sized `[batch_size, num_hidden]`.
            w_dec_i: The decoder weight terms for the dimension,
                sized `[num_hidden, 1]`.
            b_dec_i: The decoder bias terms, sized `[batch_size, 1]`.

        Returns:
            cond_p_i: The conditional probability of the dimension,
                sized `[batch_size, 1]`.
            cond_l_i: The conditional logits of the dimension,
                sized `[batch_size, 1]`.
        """
        # Decode hidden units to get conditional probability.
        h = tf.sigmoid(a)
        cond_l_i = b_dec_i + tf.matmul(h, w_dec_i)
        cond_p_i = tf.sigmoid(cond_l_i)
        return cond_p_i, cond_l_i
