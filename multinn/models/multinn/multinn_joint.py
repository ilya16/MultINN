"""Implementation of a Joint MultINN model without per-track modules."""

import tensorflow as tf

from models.multinn.core.multinn_core import MultINNCore
from utils.sequences import flatten_maybe_padded_sequences


class MultINNJoint(MultINNCore):
    """Joint MultINN model.

    One Encoder + One Generator.
    No per-track modules.

    Works with stacked multi-track sequences and learn joint
    distributions of input features.
    """

    def __init__(self, config, params, name='MultINN-joint'):
        """Initializes Joint MultiNN model.

        Args:
            config: A dict of the experiment configurations.
            params: A dict of the model parameters.
            name: Optional model name to use as a prefix when adding operations.
        """
        super().__init__(config, params, name=name)
        self._mode = 'joint'

    def _init_encoders(self, encoder_class):
        """Initializes Encoder of the Joint MultINN model.

        One joint Encoder for all tracks.

        Args:
            encoder_class: A Python class of the Encoder.

        Returns:
            encoders: A list with one Encoder.
        """
        num_dims = self.num_dims * self.num_tracks

        encoders = [
            encoder_class(
                num_dims=num_dims,
                num_hidden=self._params['encoder']['num_hidden'],
                track_name='all')
        ]
        self._encoder = encoders[0]
        self._num_dims_generator = self._encoder.num_hidden[-1]

        return encoders

    def _init_generators(self, generator_class):
        """Initializes Generator of the Joint MultINN model.

        One joint Generator for all tracks.

        Args:
            generator_class: A Python class of the Generator.

        Returns:
            generators: A list with one Generator.
        """
        generators = [
            generator_class(
                num_dims=self._num_dims_generator,
                num_hidden=self._params['generator']['num_hidden'],
                num_hidden_rnn=self._params['generator']['num_hidden_rnn'],
                keep_prob=self.keep_prob)
        ]
        self._generator = generators[0]

        return generators

    def _build_inputs(self):
        """Prepares inputs to the Joint MultINN model.

        Returns:
            inputs: A batch of stacked track input sequences.
        """
        # Combining all tracks into one sequence
        inputs = tf.reshape(
            self._x,
            tf.concat([tf.shape(self._x)[:2], [self.num_dims * self.num_tracks]], 0)
        )

        # Zero padding the first time step
        return tf.pad(inputs, [[0, 0], [1, 0], [0, 0]])

    def _build_targets(self):
        """Builds prediction targets from inputs.

        Returns:
            inputs: A batch of stacked track target predictions.
        """
        return tf.reshape(
            flatten_maybe_padded_sequences(self._x, self._lengths),
            [-1, self.num_dims * self.num_tracks]
        )

    def _build_encoders(self, mode='eval'):
        """Building the Joint MultINN Encoder.

        Args:
            Build mode for optimizing the graph size.
        """
        self._encoder.build(self._inputs, self._lengths, mode=mode)

    def _encode_inputs(self):
        """Encodes the inputs through the Joint MultINN Encoder.

        Returns:
            encoded_inputs: A batch of encoded input sequences,
                the inputs to the Joint MultINN Generator.
        """
        _, x_encoded = self._encoder.encode()

        if not self.tune_encoder:
            x_encoded = tf.stop_gradient(x_encoded)

        return x_encoded

    def _build_generators(self, mode='eval'):
        """Building the Joint MultINN Generator.

        Inputs to the Generator are outputs from the Encoder.

        Args:
            Build mode for optimizing the graph size.
        """
        with tf.variable_scope(f'generator_inputs'):
            generator_inputs = self._x_encoded[:, 0:-1, :]

        with tf.variable_scope(f'generator_targets'):
            generator_targets = self._x_encoded[:, 1:, :]

        self._generator.build(x=generator_inputs, y=generator_targets,
                              lengths=self._lengths, is_train=self._is_train, mode=mode)

    def _build_generator_outputs(self):
        """Returns the outputs from the Joint MultINN Generator.

        Returns:
            generator_outputs: Flattened stacked outputs from the Generator.
        """
        return self._generator.forward()

    def _decode_generator_outputs(self):
        """Decodes the Generators' outputs through the Encoders.

        Returns:
            cond_probs: Flattened conditional probabilities for the decoded outputs.
            decoded_outputs: Flattened decoded generator outputs.
        """
        cond_probs, decodings = self._encoder.decode(self._x_hidden)
        return cond_probs, decodings

    def build_metrics(self, targets, predictions, cond_probs=None, log_probs=None):
        """Builds Joint MultINN model's metrics.

        Args:
            targets: Flattened stacked target predictions for each track,
                sized `[batch_size, num_output_dims]`.
            predictions: Flattened stacked model's predictions for each track,
                sized `[batch_size, num_output_dims]`.
            cond_probs: Flattened stacked conditional probabilities for predictions,
                sized `[batch_size, num_output_dims]`.
            log_probs: Log probabilities for predictions,
                sized `[batch_size]`.

        Returns:
            metrics: A dictionary of metric names and metric ops.
            metrics_upd: A list of metric update ops.
            metric_summaries: A tf.Summary object for model's metrics.
        """
        metrics, metrics_upd, summaries = self._encoder.build_metrics(
            targets=targets, predictions=predictions, cond_probs=cond_probs
        )

        # Computing mean
        metrics['batch/loss'] /= self.num_tracks
        metrics['log_likelihood'] /= self.num_tracks
        metrics['perplexity'] /= self.num_tracks

        return metrics, metrics_upd, summaries

    def generate(self, num_steps):
        """Generating new sequences with the Joint MultINN.

        The generative process starts by encoding the batch of input sequences
        and going through them to obtain the final inputs' state.
        Then, the Joint Generator generates `num_steps` new steps by sampling
        from the modelled conditional distributions.
        Finally, samples are decoded through the Joint Encoder to get the
        samples in original multi-track input format.

        Args:
            num_steps: A number of time steps to generate.

        Returns:
            samples: A batch of generated sequences,
                sized `[batch_size, num_steps, num_dims, num_tracks]`
        """
        # Generator forward pass
        samples_h = self._generator.generate(self._x_encoded, num_steps)

        # Decoding inputs into the original format
        with tf.variable_scope(f'samples/'):
            _, samples = self._encoder.decode(samples_h)

            # Making the samples multi-track
            music = tf.reshape(samples, [-1, num_steps, self.num_dims, self.num_tracks], name='music')

        return music

    def train_encoders(self, optimizer, lr, layer=0):
        """Constructs ops for training Joint MultINN Encoder.

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
        return self._encoder.train(optimizer, lr)

    def pretrain_generators(self, optimizer, lr, separate_losses=True):
        """Constructs training ops for pre-training Joint MultINN Generator.

        Args:
            optimizer: tf.Optimizer object that computes gradients.
            lr: A learning rate for weight updates.
            separate_losses: `bool` indicating whether to optimize
                the separate losses for each track or optimize
                the average loss for all tracks.

        Returns:
            init_ops: A list of weight initialization ops.
            update_ops: A list of weights update ops.
            metrics: A dictionary of metric names and metric ops.
            metrics_upd: A list of metric update ops.
            summaries: A dict of tf.Summary objects for model's
                metrics, weights and training gradients.
        """
        init_ops, update_ops, metrics, metrics_upd, summaries = self._generator.pretrain(optimizer, lr)

        # Combining with summaries on the encoder level
        summaries['metrics'] = tf.summary.merge([self.summaries['metrics'], summaries['metrics']])
        metrics_upd = self.metrics_upd + metrics_upd
        metrics['global'] = self.metrics

        return init_ops, update_ops, metrics, metrics_upd, summaries

    def train_generators(self, optimizer, lr, separate_losses=True):
        """Constructs training ops for training Joint MultINN Generator.

        Args:
            optimizer: tf.Optimizer object that computes gradients.
            lr: A learning rate for weight updates.
            separate_losses: `bool` indicating whether to optimize
                the separate losses for each track or optimize
                the average loss for all tracks.

        Returns:
            init_ops: A list of weight initialization ops.
            update_ops: A list of weights update ops.
            metrics: A dictionary of metric names and metric ops.
            metrics_upd: A list of metric update ops.
            summaries: A dict of tf.Summary objects for model's
                metrics, weights and training gradients.
        """
        init_ops, update_ops, metrics, metrics_upd, summaries = self._generator.train(optimizer, lr)

        # Combining with summaries on encoder level
        summaries['metrics'] = tf.summary.merge([self.summaries['metrics'], summaries['metrics']])
        metrics_upd = self.metrics_upd + metrics_upd
        metrics['global'] = self.metrics

        return init_ops, update_ops, metrics, metrics_upd, summaries

    @property
    def trainable_feedback_variables(self):
        """An empty list of the Joint MultINN trainable feedback module variables."""
        return []
