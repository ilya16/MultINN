"""Implementation of a Composer MultINN model with shared temporal layers."""

import tensorflow as tf

from models.generators.rnn_multinade import RnnMultiNADE
from models.multinn.core.multi_encoder_nn import MultIEncoderNN
from utils.auxiliary import get_current_scope


class MultINNComposer(MultIEncoderNN):
    """Composer MultINN model.

    Multiple per-track Encoders
    + One Generator with a shared temporal unit
    and multiple distribution estimators, one per track.

    Works with multi-track sequences and learns the inter-track relations
    through the shared temporal unit.
    """

    def __init__(self, config, params, name='MultINN-composer'):
        """Initializes Composer MultiNN model.

        Args:
            config: A dict of the experiment configurations.
            params: A dict of the model parameters.
            name: Optional model name to use as a prefix when adding operations.
        """
        super().__init__(config, params, name=name)
        self._mode = 'composer'

    def _init_generators(self, generator_class):
        """Initializes Generator of the Composer MultINN model.

        One Generator with multiple estimators for all tracks.

        Args:
            generator_class: A Python class of encoders.

        Returns:
            generators: A list with one Generator.
        """
        # one generator with shared temporal unit
        if self.generator_type == 'RBM':
            raise NotImplementedError("MultiRNNRBM is not implemented yet :(")
        elif self.generator_type == 'NADE':
            generator_class = RnnMultiNADE

        generators = [
            generator_class(
                num_dims=self._num_dims_generator,
                num_hidden=self._params['generator']['num_hidden'],
                num_hidden_rnn=self._params['generator']['num_hidden_rnn'],
                tracks=self.tracks,
                keep_prob=self.keep_prob)
        ]
        self._generator = generators[0]

        return generators

    def _build_generators(self, mode='eval'):
        """Building the Composer MultINN Generator.

        Composer MultINN Generator consists of a shared temporal unit
        and per-track distribution estimators, one for each track.

        Inputs to the Generator are the stacked outputs from the Encoders.

        Args:
            Build mode for optimizing the graph size.
        """
        # Stacking encoded inputs
        with tf.variable_scope(f'{get_current_scope()}encodings/stacked/'):
            x_encoded_stack = tf.stack(self._x_encoded, axis=3)
            self._x_encoded_stack = tf.reshape(
                x_encoded_stack,
                tf.concat([tf.shape(x_encoded_stack)[:2], [self._num_dims_generator * self.num_tracks]], 0)
            )

        with tf.variable_scope(f'generator_inputs'):
            generator_inputs = self._x_encoded_stack[:, :-1, :]

        with tf.variable_scope(f'generator_targets'):
            generator_targets = self._x_encoded_stack[:, 1:, :]

        self._generator.build(x=generator_inputs, y=generator_targets,
                              lengths=self._lengths, is_train=self._is_train, mode=mode)

    def _build_generator_outputs(self):
        """Returns the outputs from the Composer MultINN Generator.

        Returns:
             generator_outputs: A list of flattened per-track outputs from the Generator.
        """
        return self._generator.forward()

    def _decode_generator_outputs(self):
        """Decodes the Generators' outputs through the Encoders.

        Returns:
            cond_probs: A list of flattened per-track conditional probabilities
                for the decoded outputs.
            decoded_outputs: A list of flattened per-track decoded generator outputs.
        """
        outputs_probs, outputs = [], []

        for i in range(self.num_tracks):
            cond_probs, decodings = self.encoders[i].decode(self._x_hidden[i])
            outputs_probs.append(cond_probs)
            outputs.append(decodings)

        return outputs_probs, outputs

    def generate(self, num_steps):
        """Generating new sequences with the Composer MultINN.

        The generative process starts by encoding the batch of input sequences
        and going through them to obtain the final inputs' state.
        Then, the Composer Generator generates `num_steps` new steps by sampling
        from the modelled conditional distributions.
        Finally, samples are decoded through the per-track Encoders to get the
        samples in original multi-track input format.

        Args:
            num_steps: A number of time steps to generate.

        Returns:
            samples: A batch of generated sequences,
                sized `[batch_size, num_steps, num_dims, num_tracks]`
        """
        # Generator forward pass
        samples_h = self._generator.generate(self._x_encoded_stack, num_steps)

        with tf.variable_scope('samples/encoded/'):
            samples_h = tf.reshape(
                samples_h,
                [tf.shape(samples_h)[0], num_steps, self._num_dims_generator, self.num_tracks]
            )
            samples_h = tf.unstack(samples_h, axis=-1)

        music = []
        # Decoding inputs into the original format
        for i in range(self.num_tracks):
            with tf.variable_scope(f'samples/{self.tracks[i]}/'):
                _, samples_i = self.encoders[i].decode(samples_h[i])

            music.append(samples_i)

        # Making the samples multi-track
        with tf.variable_scope('samples/'):
            return tf.stack(music, axis=3, name='music')

    def pretrain_generators(self, optimizer, lr, separate_losses=False):
        """Constructs training ops for pre-training Composer MultINN Generator.

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
        return self._generator.pretrain(optimizer, lr)

    def train_generators(self, optimizer, lr, separate_losses=False):
        """Constructs training ops for training MultINN Generators.

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
        init_ops, update_ops, metrics, metrics_upd, summaries = self._generator.train(optimizer, lr)

        # Combining with summaries on Encoder level
        summaries['metrics'] = tf.summary.merge([self.summaries['metrics'], summaries['metrics']])
        metrics_upd = self.metrics_upd + metrics_upd
        metrics['global'] = self.metrics

        return init_ops, update_ops, metrics, metrics_upd, summaries

    @property
    def trainable_feedback_variables(self):
        """An empty list of the Composer MultINN trainable feedback module variables."""
        return []
