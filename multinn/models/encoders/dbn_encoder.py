"""Implementation of a DBN Encoder."""

import tensorflow as tf

from models.common.dbn import DBN
from models.encoders.encoder import Encoder
from utils.auxiliary import get_current_scope
from utils.sequences import reshape_to_batch_time, flatten_maybe_padded_sequences, \
    flatten_sequences_and_save_shape


class DBNEncoder(Encoder):
    """DBN Encoder that uses binary DBN to encode and decode inputs.

    Since DBN consists of multiple RBM layers,
    a RBM Encoder can be represented as DBN Encoder with single layer.
    """

    def __init__(self,
                 num_dims,
                 num_hidden,
                 k=2,
                 name='dbn-encoder',
                 track_name='all'):
        """Initializes PassEncoder.

        Args:
            num_dims: The number of input dimensions of the DBNEncoder.
            num_hidden: The number of hidden units of the DBNEncoder (=RBMEncoder),
                or a list of hidden layer units.
            name: The model name to use as a prefix when adding operations.
            track_name: The track name on which encoder operates.
        """

        super().__init__(
            num_dims=num_dims, num_hidden=num_hidden,
            name=name, track_name=track_name
        )

        # Initializing the DBN module.
        with tf.name_scope(f'{get_current_scope()}{self.name}/{self.track_name}/'):
            self.dbn = DBN(
                num_dims=self.num_dims,
                num_hidden=self.num_hidden,
                k=k
            )

        # DBNEncoder variables
        self._variables = self.dbn.variables
        self._trainable_variables = self.dbn.trainable_variables

    def build(self, x=None, y=None, lengths=None, is_train=None, mode='eval'):
        """Builds DBNEncoder's graph.

        Defines the forward and reconstruct passes saving the encodings
        and decodings at each RBM layer.

        Args:
            x: A batch of input sequences passed to the encoder,
                sized `[batch_size, time_steps, num_dims]`.
            y: Optional targets for the encoder,
                sized `[batch_size, time_steps, num_hidden[-1]]`.
            lengths: The lengths of input sequences sized `[batch_size]`.
            is_train: (Unused) `bool` for distinguishing between training and inference.
            mode: Build mode for optimizing the graph size.
                Some operations may not be needed for training or inference.

        Raises:
            ValueError: If incorrect build mode is provided.
        """
        super().build(x, y, lengths, is_train, mode)

        if mode in ('train', 'eval'):
            with tf.variable_scope(f'{self.name}/{self.track_name}'):
                # Processing inputs
                with tf.variable_scope('inputs'):
                    self._inputs = x
                    batch_time_shape = tf.shape(x)[:2]
                    x = tf.reshape(x, [-1, self.num_dims])
                    targets = x

                # Forward pass through `self.num_layers` layers
                with tf.variable_scope('encodings'):
                    for i in range(self.dbn.num_layers):
                        with tf.variable_scope(str(i)):
                            p_x, x = self.dbn.rbm_layers[i].forward(x)
                            self._enc_probs.append(reshape_to_batch_time(p_x, batch_time_shape))
                            self._encodings.append(reshape_to_batch_time(x, batch_time_shape))

                # Backward (reconstruct) pass through `self.num_layers` layers
                with tf.variable_scope('decodings'):
                    for i in range(self.dbn.num_layers - 1, -1, -1):
                        with tf.variable_scope(str(i)):
                            p_x, x = self.dbn.rbm_layers[i].reconstruct(x)
                            self._dec_probs = [reshape_to_batch_time(p_x, batch_time_shape)] + self._dec_probs
                            self._decodings = [reshape_to_batch_time(x, batch_time_shape)] + self._decodings

            # Building model metrics and summaries for the decoded outputs
            self._metrics, self._metrics_upd, metrics_summaries = self.build_metrics(
                targets=targets, predictions=x, cond_probs=p_x
            )

            self._summaries['metrics'] = metrics_summaries
            self._summaries['weights'] = self.weight_summary

        self._is_built = True

    def build_metrics(self, targets, predictions, cond_probs=None, log_probs=None, layer=None):
        """Builds DBNEncoder metrics ops based on predictions and target values.

        Uses internal DBN to compute the metrics.

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
            metrics, metrics_upd, summaries = self.dbn.build_metrics(targets, predictions, cond_probs, log_probs)

        return metrics, metrics_upd, summaries

    def encode(self, x=None):
        """Encodes inputs through the DBN.

        Each encoder implementation should define the function explicitly.

        Args:
            x: A batch of inputs to the DBNEncoder,
                sized `[batch_size, time_steps, num_dims]` or `[batch_size, num_dims]`,
                or None if `self.encodings` from the graph construction should be used.

        Returns:
            p_h: The conditional probabilities for encodings on the last hidden layer,
                sized `[batch_size, num_hidden[-1]]`.
            h: A batch of encoded outputs,
                sized `[batch_size, num_hidden[-1]]`.
        """
        if x is None:
            return self.enc_probs[-1], self.encodings[-1]

        x, batch_time_shape = flatten_sequences_and_save_shape(x)

        p_h, h = self.dbn.forward(x)

        p_h = reshape_to_batch_time(p_h, batch_time_shape)
        h = reshape_to_batch_time(h, batch_time_shape)

        return p_h, h

    def decode(self, h=None):
        """Decodes hidden vectors through the DBN.

        Each encoder implementation should define the function explicitly.

        Args:
            h: A batch of hidden vector inputs to the Encoder,
                sized `[batch_size, time_steps, num_hidden[-1]]` or `[batch_size, num_hidden[-1]]`,
                or None if `self.encodings` from the graph construction should be used.

        Returns:
            p_h: The conditional probabilities for decodings on the input layer,
                sized `[batch_size, num_dims]`.
            h: A batch of decoded outputs,
                sized `[batch_size, num_dims]`.
        """
        if h is None:
            return self.dec_probs[-1], self.decodings[-1]

        h, batch_time_shape = flatten_sequences_and_save_shape(h)

        p_x, x = self.dbn.reconstruct(h)

        p_x = reshape_to_batch_time(p_x, batch_time_shape)
        x = reshape_to_batch_time(x, batch_time_shape)

        return p_x, x

    def train(self, optimizer, lr, layer=0):
        """Constructs training ops for the DBNEncoder.

        Args:
            optimizer: tf.Optimizer object that computes gradients.
            lr: A learning rate for weight updates.
            layer: A layer index for which the parameters should be updated.

        Returns:
            init_ops: A list of weight initialization ops.
            update_ops: A list of weights update ops.
            metrics: A dictionary of metric names and metric ops.
            metrics_upd: A list of metric update ops.
            summaries: A dict of tf.Summary objects for model's
                metrics, weights and training gradients.
        """
        assert 0 <= layer < self.num_layers

        # RBM to be trained
        rbm = self.dbn.rbm_layers[layer]

        # Accessing inputs to the `layer`-th RBM.
        x = self._inputs if layer == 0 else self.encodings[layer - 1]
        x = flatten_maybe_padded_sequences(x, self._lengths)

        # Building hidden vectors and reconstructions for the given RBM.
        with tf.variable_scope(f'{rbm.name}-{layer}/{rbm.track_name}'):
            with tf.variable_scope('encodings'):
                _, encodings = rbm.forward(x)

            with tf.variable_scope('decodings'):
                dec_probs, decodings, = rbm.reconstruct(encodings)

        # Building metrics and summaries
        metrics, metrics_upd, metrics_summaries = rbm.build_metrics(
            targets=x,
            predictions=decodings,
            cond_probs=dec_probs
        )

        summaries = {
            'weights': rbm.weight_summary,
            'metrics': metrics_summaries,
        }

        # Getting update ops
        init_ops, update_ops, summaries['gradients'] = rbm.train(x, lr)

        return init_ops, update_ops, metrics, metrics_upd, summaries
