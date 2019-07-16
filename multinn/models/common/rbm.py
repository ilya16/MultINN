"""Implementation of a RBM (Restricted Boltzmann Machine)."""

import tensorflow as tf
import tensorflow_probability as tfp

from metrics.statistical import base_metrics, build_summaries
from models.common.model import Model
from utils.auxiliary import safe_log


class RBM(Model):
    """Restricted Boltzmann Machine [1].

    [1]: https://www.csrc.ac.cn/upload/file/20170703/1499052743888438.pdf
    """

    def __init__(self,
                 num_dims,
                 num_hidden=128,
                 k=10,
                 name='rbm'):
        """Initializes RBM.

        Args:
            num_dims: The number of input/output dimensions of the RBM.
            num_hidden: The number of hidden units of the RBM.
            k: The number of steps to use in Gibbs sampling.
            name: The name for the RBM model.
        """
        super().__init__(name=name)

        self._num_dims = num_dims
        self._num_hidden = num_hidden
        self._k = k

        initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope(self.name):
            # RBM weight matrix.
            self.W = tf.Variable(
                initializer([self.num_dims, self.num_hidden]),
                dtype=tf.float32,
                name="W"
            )

            # Hidden layer bias term.
            self.bh = tf.Variable(
                tf.zeros([1, self.num_hidden], tf.float32),
                dtype=tf.float32,
                name="bh"
            )

            # Visible layer bias term.
            self.bv = tf.Variable(
                tf.zeros([1, self.num_dims], tf.float32),
                dtype=tf.float32,
                name="bv"
            )

        # RBM variables
        self._variables = {'W': self.W, 'bh': self.bh, 'bv': self.bv}
        self._trainable_variables = [self.W, self.bv, self.bh]

        self.build()

    @property
    def num_dims(self):
        """int: The number of input/output dimensions of the RBM."""
        return self._num_dims

    @property
    def num_hidden(self):
        """int: The number of hidden units of the RBM."""
        return self._num_hidden

    @property
    def k(self):
        """The number of Gibbs sampling steps (the length of Gibbs chain)."""
        return self._k

    def build(self, x=None, y=None, lengths=None, is_train=None, mode='eval'):
        """Builds RBM's Graph.

        Since all variables of RBM are defined during the model initialization
        and RBM is a base model mostly used by other models that define
        the operation graph, the `build()` function only updates the conventional
        variable `is_built` and is called inside `__init__()`.

        Raises:
            ValueError: If incorrect build mode is provided.
        """
        super().build(x, y, lengths, is_train, mode)

        self._is_built = True

    def build_metrics(self, targets, predictions, cond_probs=None, log_probs=None):
        """Builds RBM metrics ops based on predictions and target values.

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
        # Computing RBM free-energy cost.
        with tf.variable_scope("cost"):
            cost, free_energy = self.free_energy_cost(targets, predictions, bh=None, bv=None)

        # Computing log probabilities from conditional probabilities.
        if log_probs is None and cond_probs is not None:
            with tf.variable_scope("log_probs"):
                log_probs = tf.losses.log_loss(
                    targets,
                    cond_probs,
                    reduction=tf.losses.Reduction.NONE
                )
                log_probs = tf.reduce_sum(log_probs, axis=1)
        else:
            raise ValueError('Incorrect arguments. Either `cond_probs`, or `log_probs` '
                             'should be provided on `rbm.build_metrics()` function call')

        # Using base model metrics.
        metrics, metrics_upd = base_metrics(cost, targets, predictions, log_probs)

        # Adding free-energy to the dictionary of metrics.
        with tf.variable_scope("free_energy_op"):
            free_energy, free_energy_upd = tf.metrics.mean(free_energy)
            metrics['free_energy'] = free_energy
            metrics_upd.append(free_energy_upd)

        # Building tf.Summary object from metrics.
        summaries = build_summaries(metrics)

        return metrics, metrics_upd, summaries

    def forward(self, v, bh=None):
        """Forward pass of inputs in RBM.

        Args:
            v: A batch of inputs to RBM, sized `[batch_size, num_dims]`.
            bh: External hidden bias terms, sized `[batch_size, num_hidden]`,
                or None if the internal bias term should be used.

        Returns:
            p_h: The activation conditional probabilities on the hidden layer,
                sized `[batch_size, num_hidden]`.
            h: The sampled hidden layer activations,
                sized `[batch_size, num_hidden]`.
        """
        bh = bh if bh is not None else self.bh

        p_h, _ = self._cond_prob_h(v, bh)
        h = self._sample(p_h)

        return p_h, h

    def reconstruct(self, h, bv=None):
        """Backward pass of hidden vectors in RBM.

        Computes reconstruction of inputs from hidden values.

        Args:
             h: A batch of hidden vector inputs, sized `[batch_size, num_hidden]`.
             bv: External visible bias terms, sized `[batch_size, num_dims]`,
                or None if the internal bias term should be used.

        Returns:
            p_v: The activation conditional probabilities on the visible layer,
                sized `[batch_size, num_dims]`.
            v: The sampled visible layer activations,
                sized `[batch_size, num_dims]`.
        """
        bv = bv if bv is not None else self.bv

        p_v, _ = self._cond_prob_v(h, bv)
        v = self._sample(p_v)

        return p_v, v

    def sample(self, v, bh=None, bv=None, k=None):
        """ Runs a k-step Gibbs chain to sample from the modelled probability distribution.

        The sampling starts from the visible layer and consists of `k` steps of
        alternating sampling from visible->hidden->visible layer from
        the conditional distributions modelled by RBM.

        Args:
            v: A batch of inputs to RBM, sized `[batch_size, num_dims]`.
                The initial vector of Gibbs chain.
            bh: External hidden bias terms, sized `[batch_size, num_hidden]`,
                or None if the internal bias term should be used.
            bv: External visible bias terms, sized `[batch_size, num_dims]`,
                or None if the internal bias term should be used.
            k: The number of Gibbs sampling steps (the length of the chain),
                or None if the internal value of `k` should be used.

        Returns:
            p_v: The sample conditional probabilities, sized `[batch_size, num_dims]`.
            v_sample: The sampled visible layer outputs, sized `[batch_size, num_dims]`.
        """

        def gibbs_step(count, p_vk, vk):
            """One step of Gibbs sampling chain."""
            p_hk, hk = self.forward(vk, bh)
            p_vk, vk = self.reconstruct(hk, bv)
            return count + 1, p_vk, vk

        # Run Gibbs chain for `k` iterations.
        ct = tf.constant(0)
        [_, p_v, v_sample] = tf.while_loop(
            lambda count, *args: tf.less(count, tf.constant(k)),
            gibbs_step,
            [ct, v, v]
        )

        # Do not propagate gradients through the chain.
        v_sample = tf.stop_gradient(v_sample, name='sample')

        return p_v, v_sample

    def free_energy_cost(self, v, v_sample, bh=None, bv=None):
        """Computes RBM's free-energy cost for target and sampled vectors.

        Free-energy cost is based on the difference between input and sample
        vectors free energies.

        Args:
            v: A batch of target visible vectors, sized `[batch_size, num_dims]`.
                The initial vector of Gibbs chain.
            v: A batch of sampled visible vectors, sized `[batch_size, num_dims]`.
                The initial vector of Gibbs chain.
            bh: External hidden bias terms, sized `[batch_size, num_hidden]`,
                or None if the internal bias term should be used.
            bv: External visible bias terms, sized `[batch_size, num_dims]`,
                or None if the internal bias term should be used.

        Returns:
            cost: A batch of free-energy cost function values, sized `[batch_size, 1]`.
            free_energy: A free-energy for a batch of inputs, sized `[batch_size, 1]`.
        """
        bh = bh if bh is not None else self.bh
        bv = bv if bv is not None else self.bv

        def F(vv):
            """Computes free-energy for a batch of inputs."""
            return -tf.reduce_sum(tf.log(1 + tf.exp(tf.matmul(vv, self.W) + bh)), 1) - tf.matmul(vv, tf.transpose(bv))

        free_energy = F(v)
        cost = tf.subtract(free_energy, F(v_sample), name='free_energy')

        return cost, free_energy

    def train(self, v, lr):
        """Constructs training ops for the RBM.

        Args:
            v: A batch of inputs to RBM, sized `[batch_size, num_dims]`.
            lr: A learning rate for weight updates.

        Returns:
            init_ops: A list of weight initialization ops.
            update_ops: A list of weights update ops.
            gradient_summaries: A tf.Summary object for monitoring gradients.
        """
        init_ops = self.visible_bias_init_ops(v)

        update_ops, gradients = self._cd_update(v, lr=lr)
        gradient_summaries = tf.summary.merge(
            [tf.summary.histogram(grad.name, grad) for grad in gradients]
        )

        return init_ops, update_ops, gradient_summaries

    def visible_bias_init_ops(self, v):
        """Initialization ops for visible bias term.
        Args:
            v: A batch of visible vectors, sized `[batch_size, num_dims]`.

        As suggested in Practical Guide to Training RBMs by Hinton in [1].
        """
        p = tf.reduce_mean(v, 0)  # proportion of training vectors in which unit i is on
        bv_ = safe_log(p / (1 - p))
        bv_ = tf.reshape(bv_, tf.shape(self.bv))

        return [self.bv.assign(bv_)]

    def _cd_update(self, v, lr):
        """Contrastive Divergence (CD) algorithm for training RBMs [2].

        [2]: https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf

        Args:
            v: A batch of inputs to RBM, sized `[batch_size, num_dims]`.
            lr: A learning rate for weight updates.

        Returns:
            update_ops: A list of weights update ops.
            gradients: A list of gradients.
        """
        # The sample of visible nodes using k-step Gibbs chain.
        p_v_sample, v_sample = self.sample(v, self.bh, self.bv, self.k)

        # The sample of the hidden nodes from the inputs.
        _, h = self.forward(v)

        # The sample of the hidden nodes from the sampled visible nodes.
        p_h_sample, h_sample = self.forward(v_sample)

        # The CD learning rate.
        lr = tf.constant(lr, tf.float32) / tf.to_float(tf.shape(v)[0])

        # Weight updates (gradients)
        W_ = tf.multiply(
            lr,
            tf.subtract(tf.matmul(tf.transpose(v), h), tf.matmul(tf.transpose(p_v_sample), p_h_sample)),
            name='W_')
        bv_ = tf.multiply(lr, tf.reduce_sum(tf.subtract(v, p_v_sample), 0, True), name='bv_')
        bh_ = tf.multiply(lr, tf.reduce_sum(tf.subtract(h, p_h_sample), 0, True), name='bh_')

        update_ops = [self.W.assign_add(W_), self.bv.assign_add(bv_), self.bh.assign_add(bh_)]
        gradients = [W_, bv_, bh_]

        return update_ops, gradients

    def _cond_prob_h(self, v, bh):
        """Computes hidden layer activation conditional probabilities.

        Args:
            v: A batch of inputs to RBM, sized `[batch_size, num_dims]`.
            bh: A hidden bias term to use, sized `[batch_size, num_hidden]`
                or `[1, num_hidden]` (broadcasting is used).

        Returns:
            cond_l_h: The activation logits on the hidden layer,
                sized `[batch_size, num_hidden]`.
            cond_p_h: The conditional probabilities on the hidden layer,
                sized `[batch_size, num_hidden]`.
        """
        cond_l_h = tf.matmul(v, self.W) + bh
        cond_p_h = tf.sigmoid(cond_l_h)

        return cond_p_h, cond_l_h

    def _cond_prob_v(self, h, bv):
        """Computes hidden layer activation conditional probabilities.

        Args:
            h: A batch of hidden vector inputs to RBM, sized `[batch_size, num_hidden]`.
            bv: A visible bias term to use, sized `[batch_size, num_dims]`
                or `[1, num_dims]` (broadcasting is used).

        Returns:
            cond_l_v: The activation logits on the visible layer,
                sized `[batch_size, num_dims]`.
            cond_p_v: The conditional probabilities on the visible layer,
                sized `[batch_size, num_dims]`.
        """
        cond_l_v = tf.matmul(h, tf.transpose(self.W)) + bv
        cond_p_v = tf.sigmoid(cond_l_v)

        return cond_p_v, cond_l_v

    @staticmethod
    def _sample(cond_probs):
        """Sampling from the Bernoulli distributions.

        Args:
            cond_probs: A batch of conditional probabilities,
                sized `[batch_size, -1]`.

        Returns:
            eval: A Bernoulli sample, sized `[batch_size, -1]`
        """
        bernoulli = tfp.distributions.Bernoulli(probs=cond_probs, dtype=tf.float32)
        return bernoulli.sample()
