"""Implementation of a Pass ("Skip") Encoder that does not alter the inputs."""

import tensorflow as tf

from metrics.statistical import base_metrics, build_summaries
from models.encoders.encoder import Encoder


class PassEncoder(Encoder):
    """Pass Encoder that does not disturb anyone and anything."""

    def __init__(self,
                 num_dims,
                 num_hidden,
                 name='pass-encoder',
                 track_name='all'):
        """Initializes PassEncoder.

        Args:
            num_dims: The number of input dimensions of the Encoder.
            num_hidden: (Unused) The number of hidden units of the Encoder.
            name: The model name to use as a prefix when adding operations.
            track_name: The track name on which encoder operates.
        """
        super().__init__(
            num_dims=num_dims, num_hidden=num_dims,
            name=name, track_name=track_name
        )

    def build(self, x=None, y=None, lengths=None, is_train=None, mode='eval'):
        """Builds PassEncoder's graph.

        Since PassEncoder does not alter the inputs, the `build()` function
        only assigns the inputs as encodings and decodings.

        Args:
            x: Optional inputs to the encoder,
                sized `[batch_size, time_steps, num_dims]`.
            y: (Unused) targets for the encoder,
                sized `[batch_size, time_steps, num_hidden[-1]]`.
            lengths: (Unused) lengths of input sequences sized `[batch_size]`.
            is_train: (Unused) `bool` for distinguishing between training and inference.
            mode: Build mode for optimizing the graph size.
                Some operations may not be needed for training or inference.

        Raises:
            ValueError: If incorrect build mode is provided.
        """
        super().build(x, y, lengths, is_train, mode)

        self._enc_probs, self._encodings = [x], [x]
        self._dec_probs, self._decodings = [x], [x]

        self._is_built = True

    def build_metrics(self, targets, predictions, cond_probs=None, log_probs=None, layer=None):
        """Builds PassEncoder metrics ops based on predictions and target values.

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
                                 'should be provided on `encoder.build_metrics()` function call')

            # Using base model metrics.
            metrics, metrics_upd = base_metrics(log_probs, targets, predictions, log_probs)

            # Building tf.Summary object from metrics.
            summaries = build_summaries(metrics)

        return metrics, metrics_upd, summaries

    def encode(self, x=None):
        """Encodes inputs through the `skip` layer.

        Technically, PassEncoder returns the inputs as encodings.

        Args:
            x: A batch of inputs to the Encoder,
                sized `[batch_size, time_steps, num_dims]` or `[batch_size, num_dims]`,
                or None if `self.encodings` from the graph construction should be used.
        """
        if x is None:
            return self.enc_probs[-1], self.encodings[-1]

        return x, x

    def decode(self, h=None):
        """Decodes hidden through the `skip` layer.

        Technically, PassEncoder returns the hidden vectors as decodings.

        Args:
            h: A batch of hidden vector inputs to the Encoder,
                sized `[batch_size, time_steps, num_hidden[-1]]` or `[batch_size, num_hidden[-1]]`,
                or None if `self.encodings` from the graph construction should be used.
        """
        if h is None:
            return self.dec_probs[-1], self.decodings[-1]

        return h, h

    def train(self, optimizer, lr, layer=0):
        """Constructs training ops for the PassEncoder.

        Since PassEncoder does not have variables and does not require training,
        the function just returns empty update ops, metrics, and summary
        lists and dictionaries.
        """
        return [], [], {}, [], {}
