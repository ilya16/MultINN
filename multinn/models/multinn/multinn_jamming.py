"""Implementation of a Jamming MultINN model with per-track modules and no communication."""

import tensorflow as tf

from models.multinn.core.multi_encoder_nn import MultIEncoderNN
from utils.training import compute_gradients


class MultINNJamming(MultIEncoderNN):
    """Jamming MultINN model.

    Multiple per-track Encoders + Multiple per-track Generators
    with no inter-communication.

    Works with multi-track sequences and learns the track specific features.

    The models with Feedback inherit from the Jamming model.
    """

    def __init__(self, config, params, name='MultINN-jamming'):
        """Initializes Jamming MultiNN model.

        Args:
            config: A dict of the experiment configurations.
            params: A dict of the model parameters.
            name: Optional model name to use as a prefix when adding operations.
        """
        super().__init__(config, params, name=name)
        self._mode = 'jamming'

    def _init_generators(self, generator_class):
        """Initializes Generators of the Jamming MultINN model.

        Args:
            generator_class: A Python class of the Generators.

        Returns:
            generators: A list of the Jamming MultINN Generators.
        """
        generators = [
            generator_class(
                num_dims=self._num_dims_generator,
                num_hidden=self._params['generator']['num_hidden'],
                num_hidden_rnn=self._params['generator']['num_hidden_rnn'],
                keep_prob=self.keep_prob,
                track_name=self.tracks[i])
            for i in range(self.num_tracks)
        ]

        return generators

    def _build_generators(self, mode='eval'):
        """Building the Jamming MultINN Generators.

        Inputs to the Generators are outputs from the Encoders.

        Args:
            Build mode for optimizing the graph size.
        """
        for i in range(self.num_tracks):
            with tf.variable_scope(f'generator_inputs/{self.tracks[i]}'):
                generator_inputs = self._x_encoded[i][:, 0:-1, :]

            with tf.variable_scope(f'generator_targets/{self.tracks[i]}'):
                generator_targets = self._x_encoded[i][:, 1:, :]

            self.generators[i].build(x=generator_inputs, y=generator_targets,
                                     lengths=self._lengths, is_train=self._is_train, mode=mode)

    def _build_generator_outputs(self):
        """Returns the outputs from the Jamming MultINN Generators.

        Returns:
             generator_outputs: A list of flattened per-track outputs from the Generators.
        """
        x_hidden = []
        for i in range(self.num_tracks):
            with tf.variable_scope(f'{self.tracks[i]}'):
                x_hidden.append(self.generators[i].forward())

        return x_hidden

    def _decode_generator_outputs(self):
        """Decodes the Generators' outputs through the Encoders.

        Returns:
            cond_probs: A list of flattened per-track conditional probabilities
                for the decoded outputs.
            decoded_outputs: A list of flattened per-track decoded generator outputs.
        """
        outputs_probs, outputs = [], []

        for i in range(self.num_tracks):
            with tf.variable_scope(f'{self.tracks[i]}'):
                cond_probs, decodings = self.encoders[i].decode(self._x_hidden[i])
                outputs_probs.append(cond_probs)
                outputs.append(decodings)

        return outputs_probs, outputs

    def generate(self, num_steps):
        """Generating new sequences with the Jamming MultINN.

        The generative process starts by encoding the batch of input sequences
        and going through them to obtain the final inputs' state.
        Then, the Jamming per-track Generators generate `num_steps` new steps
        by sampling from the modelled conditional distributions.
        Finally, samples are decoded through the per-track Encoders to get the
        samples in original multi-track input format.

        Args:
            num_steps: A number of time steps to generate.

        Returns:
            samples: A batch of generated sequences,
                sized `[batch_size, num_steps, num_dims, num_tracks]`
        """
        music = []

        for i in range(self.num_tracks):
            # Generator forward pass
            with tf.variable_scope(f'{self.tracks[i]}'):
                samples_h = self.generators[i].generate(self._x_encoded[i], num_steps)

            # Decoding inputs into the original format
            with tf.variable_scope(f'samples/{self.tracks[i]}/'):
                _, samples = self.encoders[i].decode(samples_h)

            music.append(samples)

        # Making the samples multi-track
        with tf.variable_scope('samples/'):
            return tf.stack(music, axis=3, name='music')

    def pretrain_generators(self, optimizer, lr, separate_losses=False):
        """Constructs training ops for pre-training per-track MultINN Generators.

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
        # Pre-train each Generator separately but in parallel
        return self._train_generators(optimizer, lr, pretrain=True, separate_losses=separate_losses)

    def train_generators(self, optimizer, lr, separate_losses=False):
        """Constructs training ops for training per-track MultINN Generators.

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
        # Training generators by minimizing mean track loss
        init_ops, update_ops, metrics, metrics_upd, summaries = self._train_generators(
            optimizer, lr, pretrain=False, separate_losses=separate_losses
        )

        # Combining with summaries on Encoder level
        summaries['metrics'] = tf.summary.merge([self.summaries['metrics'], summaries['metrics']])
        metrics_upd = self.metrics_upd + metrics_upd
        metrics['global'] = self.metrics

        return init_ops, update_ops, metrics, metrics_upd, summaries

    def _train_generators(self, optimizer, lr, pretrain=False, separate_losses=False):
        """Auxiliary function for constructing training ops for MultINN Generators.

        Combines the functionality needed to build pre-training and training ops.

        Args:
            optimizer: tf.Optimizer object that computes gradients.
            lr: A learning rate for weight updates.
            pretrain: `bool` indicating whether to pre-train or train
                the Generators.
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
        # (Pre-)training Fenerators by minimizing mean track loss
        init_ops, update_ops = [], []
        track_metrics, track_metrics_upd, track_summaries = [], [], []

        # Collecting per-track metrics
        for i in range(self.num_tracks):
            if pretrain:
                train_func = self.generators[i].pretrain
            else:
                train_func = self.generators[i].train

            init_ops_i, update_ops_i, metrics_i, metrics_upd_i, summaries_i = train_func(
                optimizer, lr, run_optimizer=separate_losses
            )

            init_ops += init_ops_i
            update_ops += update_ops_i
            track_metrics.append(metrics_i)
            track_metrics_upd.append(metrics_upd_i)
            track_summaries.append(summaries_i)

        # Combining metrics and summaries
        metrics, metrics_upd, summaries = self._combine_track_metrics(
            track_metrics, track_metrics_upd, track_summaries,
            global_scope=f"metrics/{self.generators[0].name}/global/"
        )

        if not separate_losses:
            # Optimizing the mean track loss
            trainable_variables = self.trainable_generator_variables + self.trainable_feedback_variables
            update_ops, summaries['gradients'] = compute_gradients(
                optimizer, metrics['batch/loss'], trainable_variables
            )
            summaries['weights'] = tf.summary.merge(
                [tf.summary.histogram(var.name, var) for var in trainable_variables]
            )

        return init_ops, update_ops, metrics, metrics_upd, summaries

    @property
    def trainable_feedback_variables(self):
        """An empty list of the Jamming MultINN trainable feedback module variables."""
        return []
