"""Implementation of an interface for the MultINN models with per-track Encoders."""

import abc

import tensorflow as tf

from models.multinn.core.multinn_core import MultINNCore
from utils.sequences import flatten_maybe_padded_sequences


class MultIEncoderNN(MultINNCore):
    """Abstract class for MultINN models with multiple per-track encoders.

    Defines only the functionality of the Encoders.
    Each child class should define the functionality of the Generators.
    """

    def __init__(self, config, params, name='MultIEncoderNN'):
        """Initializes MultIEncoderNN.

        Args:
            config: A dict of the experiment configurations.
            params: A dict of the model parameters.
            name: Optional model name to use as a prefix when adding operations.
        """
        super().__init__(config, params, name=name)

    def _init_encoders(self, encoder_class):
        """Initializes per-track Encoders of the MultIEncoderNN models.

        One Encoder for each track.

        Args:
            encoder_class: A Python class of encoders.

        Returns:
            encoders: A list of the MultINN Encoders.
        """
        num_dims = self.num_dims

        encoders = [
            encoder_class(
                num_dims=num_dims,
                num_hidden=self._params['encoder']['num_hidden'],
                track_name=self.tracks[i])
            for i in range(self.num_tracks)
        ]
        self._num_dims_generator = encoders[0].num_hidden[-1]

        return encoders

    @abc.abstractmethod
    def _init_generators(self, generator_class):
        """Initializes Generators of the MultIEncoderNN models.

        Each MultIEncoderNN model implementation should define the function explicitly.

        Args:
            generator_class: A Python class of generators.

        Returns:
            generators: A list of the MultINN Generators.
        """
        pass

    def _build_inputs(self):
        """Prepares inputs to the MultIEncoderNN models.

        Returns:
            inputs: A list of per-track input sequences to the MultINN model.
        """
        # Zero padding the first time step
        inputs = tf.pad(self._x, [[0, 0], [1, 0], [0, 0], [0, 0]], name='padded_inputs')

        # Each encoder works with a separate input sequence
        return tf.unstack(inputs, axis=-1)

    def _build_targets(self):
        """Builds prediction targets from inputs.

        Returns:
            inputs: A list of per-track target predictions.
        """
        targets_flat = flatten_maybe_padded_sequences(self._x, self._lengths)

        # Transforming into a list of per-track predictions
        return tf.unstack(targets_flat, axis=-1)

    def _build_encoders(self, mode='eval'):
        """Building the MultIEncoderNN Encoders.

        Args:
            Build mode for optimizing the graph size.
        """
        for i in range(self.num_tracks):
            self.encoders[i].build(self._inputs[i], lengths=self._lengths, mode=mode)

    def _encode_inputs(self):
        """Encodes the inputs through the MultIEncoderNN Encoders.

        Returns:
            encoded_inputs: A list of of per-track encoded input sequences,
                the inputs to the MultINN Generators.
        """
        x_encoded = []

        # Encoding inputs for each track
        for i in range(self.num_tracks):
            with tf.variable_scope(f'{self.tracks[i]}'):
                _, x_encoded_i = self.encoders[i].encode()
                if not self.tune_encoder:
                    x_encoded_i = tf.stop_gradient(x_encoded_i)
                x_encoded.append(x_encoded_i)

        return x_encoded

    def build_metrics(self, targets, predictions, cond_probs=None, log_probs=None):
        """Builds MultIEncoderNN models' metrics on the global level.

        Args:
            targets: A list of flattened target predictions for each track,
                sized `[batch_size, num_output_dims]`.
            predictions: A list of flattened model's predictions for each track,
                sized `[batch_size, num_output_dims]`.
            cond_probs: A list of flattened conditional probabilities for predictions,
                sized `[batch_size, num_output_dims]`.
            log_probs: A list of log probabilities for predictions,
                sized `[batch_size]`.

        Returns:
            metrics: A dictionary of metric names and metric ops.
            metrics_upd: A list of metric update ops.
            metric_summaries: A tf.Summary object for model's metrics.
        """
        track_metrics, track_metrics_upd, track_summaries = [], [], []

        # Building per-track metrics on the encoder level
        for i in range(self.num_tracks):
            metrics_i, metrics_upd_i, summaries_i = self.encoders[i].build_metrics(
                targets=targets[i], predictions=predictions[i], cond_probs=cond_probs[i]
            )

            track_metrics.append(metrics_i)
            track_metrics_upd.append(metrics_upd_i)
            track_summaries.append({'metrics': summaries_i})

        # Combining per-track metrics and summaries
        metrics, metrics_upd, summaries = self._combine_track_metrics(
            track_metrics, track_metrics_upd, track_summaries,
            global_scope=f"metrics/{self.encoders[0].name}/global/"
        )

        return metrics, metrics_upd, summaries

    def train_encoders(self, optimizer, lr, layer=0):
        """Constructs ops for training per-track MultINN Encoders.

        Args:
            optimizer: tf.Optimizer object that computes gradients.
            lr: A learning rate for weight updates.
            layer: A layer index for which the parameters should be updated,
                or None if the the whole Encoder is trained.

        Returns:
            init_ops: A list of weight initialization ops.
            update_ops: A list of weights update ops.
            metrics: A dictionary of metric names and metric ops.
            metrics_upd: A list of metric update ops.
            summaries: A dict of tf.Summary objects for model's
                metrics, weights and training gradients.
        """
        # Pre-train each Encoder separately but in parallel
        init_ops, update_ops = [], []
        track_metrics, track_metrics_upd, track_summaries = [], [], []

        # Building per-track metrics and summaries
        for i in range(self.num_tracks):
            init_ops_i, update_ops_i, metrics_i, metrics_upd_i, summaries_i = self.encoders[i].train(
                optimizer, lr, layer=layer
            )

            init_ops += init_ops_i
            update_ops += update_ops_i
            track_metrics.append(metrics_i)
            track_metrics_upd.append(metrics_upd_i)
            track_summaries.append(summaries_i)

        # Combining per-track metrics and summaries
        layer_str = '' if layer == 0 else f'/{layer}'
        scope = f"metrics/{self.encoders[0].name}{layer_str}/global/"
        metrics, metrics_upd, summaries = self._combine_track_metrics(
            track_metrics, track_metrics_upd, track_summaries, global_scope=scope
        )

        return init_ops, update_ops, metrics, metrics_upd, summaries
