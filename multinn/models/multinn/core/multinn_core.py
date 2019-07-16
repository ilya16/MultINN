"""Implementation of a core functionality of the MultINN models."""

import abc
import itertools
import os

import tensorflow as tf

from metrics.musical_tf import get_metric_summary_ops
from models.encoders.dbn_encoder import DBNEncoder
from models.encoders.pass_encoder import PassEncoder
from models.generators.rnn_nade import RnnNade
from models.generators.rnn_rbm import RnnRBM
from models.multinn.core.multinn_interface import MultINNInterface


class MultINNCore(MultINNInterface):
    """MultINN Core.

    Defines the core variables and implements the functionality
    used by all MultINN model implementations.
    """

    def __init__(self, config, params, name='MultINN'):
        """Initializes MultINN Core.

        Args:
            config: A dict of the experiment configurations.
            params: A dict of the model parameters.
            name: Optional model name to use as a prefix when adding operations.
        """
        super().__init__(name=name)

        self._mode = 'core'
        self._config = config
        self._params = params
        self._encoder_type = params['encoder']['type']
        self._generator_type = params['generator']['type']

        # Getting Encoders' class
        if self.encoder_type == 'Pass':
            encoder_class = PassEncoder
        elif self.encoder_type in ('RBM', 'DBN'):
            encoder_class = DBNEncoder
        else:
            raise ValueError('Incorrect encoder type, supported types are `Pass`, `RBM`, and `DBN`')

        # Getting Generators' class
        if self.generator_type == 'RBM':
            generator_class = RnnRBM
        elif self.generator_type == 'NADE':
            generator_class = RnnNade
        else:
            raise ValueError('Incorrect generator type, supported types are `RBM`, and `NADE`')

        # Computing the dimensionality of inputs for one track
        num_dims = config['data']['pitch_range']['highest'] - config['data']['pitch_range']['lowest']
        num_dims *= config['training']['num_pixels']
        self._num_dims = num_dims

        self._tracks = config['data']['instruments']

        self._feedback_module = False
        self._keep_prob = params['keep_prob']
        self._tune_encoder = params['tune_encoder']

        # Defining model placeholders
        with tf.variable_scope(f'{self.name}/'):
            # Inputs shape - [batch, max_time, num_dims, num_instruments]
            self._x = tf.placeholder(tf.float32, [None, None, num_dims, self.num_tracks], name='x')
            self._lengths = tf.placeholder(tf.int32, [None], 'lengths')
            self._is_train = tf.placeholder(tf.bool, name='is_train')

        self._placeholders = {
            'x': self._x,
            'lengths': self._lengths,
            'is_train': self._is_train
        }

        # Initializing main modules
        with tf.name_scope(self.name + '/'):
            self._encoders = self._init_encoders(encoder_class)
            self._generators = self._init_generators(generator_class)

        # Additional variables used by the models
        self._x_encoded = None
        self._x_hidden = None
        self._outputs_probs = None

    @abc.abstractmethod
    def _init_encoders(self, encoder_class):
        """Initializes Encoders of the MultINN model.

        Each MultINN model implementation should define the function explicitly.

        Args:
            encoder_class: A Python class of the Encoders.

        Returns:
            encoders: A list of the MultINN Encoders.
        """
        pass

    @abc.abstractmethod
    def _init_generators(self, generator_class):
        """Initializes Generators of the MultINN model.

        Each MultINN model implementation should define the function explicitly.

        Args:
            generator_class: A Python class of the Generators.

        Returns:
            generators: A list of the MultINN Generators.
        """
        pass

    @property
    def mode(self):
        """str: MultINN operation mode."""
        return self._mode

    @property
    def num_dims(self):
        """int: The number of input dimensions the models."""
        return self._num_dims

    @property
    def tracks(self):
        """list of str: The names of tracks with which MultINN operates."""
        return self._tracks

    @property
    def num_tracks(self):
        """int: The number of tracks with which MultINN operate."""
        return len(self._tracks)

    @property
    def encoder_type(self):
        """str: The model type of Encoders that MultINN utilizes."""
        return self._encoder_type

    @property
    def generator_type(self):
        """str: The model type of Generators that MultINN utilizes."""
        return self._generator_type

    @property
    def encoders(self):
        """The list of the MultINN Encoders."""
        return self._encoders

    @property
    def generators(self):
        """The list of the MultINN Generators."""
        return self._generators

    @property
    def feedback_module(self):
        """`bool` indicating whether the model utilized feedback modules."""
        return self._feedback_module

    @property
    def keep_prob(self):
        """float: Dropout keep probability."""
        return self._keep_prob

    @property
    def tune_encoder(self):
        """`bool` indicating whether to update encoders' parameters while training generators."""
        return self._tune_encoder

    @property
    def loss(self):
        """MultINN model loss op."""
        return self.metrics['batch/loss']

    def build(self, x=None, y=None, lengths=None, is_train=None, mode='eval'):
        """Builds MultINN model's graph.

        Defines the forward pass through the input sequences.

        The inputs are encoded through the Encoders and passed to Generators
        that learn the conditional probabilities of encoded vectors.

        Note:
            MultINN uses placeholders for inputs and lengths and does not
            utilize any of the passed arguments except the `mode`.

        Args:
            x: (Unused) A batch of input sequences to the MultINN,
                sized `[batch_size, time_steps, step_span, num_track]`.
            y: (Unused) A batch of target predictions for the generator,
                sized `[batch_size, time_steps, step_span, num_track]`.
            lengths: (Unused) The lengths of input sequences sized `[batch_size]`.
            is_train: (Unused) `bool` for distinguishing between training and inference.
            mode: Build mode for optimizing the graph size.
                Some operations may not be needed for training or inference.

        Raises:
            ValueError: If incorrect build mode is provided.
        """
        super().build(x, y, lengths, is_train, mode)

        with tf.name_scope(self.name + '/'):
            # Building inputs
            with tf.variable_scope('inputs'):
                self._inputs = self._build_inputs()

            # Building encoders
            self._x_encoded = self._build_encoders('eval')

            # Encoding inputs
            with tf.variable_scope('encodings'):
                self._x_encoded = self._encode_inputs()

            # Building generators
            self._build_generators()

        # Building model variables
        self._build_variables()

        with tf.name_scope(self.name + '/'):
            if mode in ('train', 'eval'):
                # Forward pass in generators
                with tf.variable_scope('generator_outputs'):
                    self._x_hidden = self._build_generator_outputs()

                # Decoding outputs into original format
                with tf.variable_scope('outputs'):
                    self._outputs_probs, self._outputs = self._decode_generator_outputs()

                # Building prediction targets
                with tf.variable_scope('targets'):
                    targets = self._build_targets()

                # Building metrics and summaries
                self._metrics, self._metrics_upd, self._summaries = self.build_metrics(
                    targets=targets, predictions=self._outputs, cond_probs=self._outputs_probs
                )

                self._summaries['weights'] = self.weight_summary

        self._is_built = True

    @abc.abstractmethod
    def _build_inputs(self):
        """Abstract function for preprocessing inputs to the MultINN model.

        Returns:
            inputs: A batch of input sequences to the MultINN model.
        """
        pass

    @abc.abstractmethod
    def _build_targets(self):
        """Abstract function for building prediction targets from inputs.

        Returns:
            inputs: A batch of target predictions.
        """
        pass

    @abc.abstractmethod
    def _build_encoders(self, mode='eval'):
        """Abstract function for building the MultINN Encoders.

        Args:
            Build mode for optimizing the graph size.
        """
        pass

    @abc.abstractmethod
    def _encode_inputs(self):
        """Abstract function for encoding the inputs through the Encoders.

        Returns:
            encoded_inputs: A batch of encoded input sequences,
                the inputs to the MultINN Generators.
        """
        pass

    @abc.abstractmethod
    def _build_generators(self, mode='eval'):
        """Abstract function for building the MultINN Generators.

        Args:
            Build mode for optimizing the graph size.
        """
        pass

    @abc.abstractmethod
    def _build_generator_outputs(self):
        """Abstract function for building the outputs from the MultINN Generators.

        Returns:
            generator_outputs: Flattened outputs from the Generators.
        """
        pass

    @abc.abstractmethod
    def _decode_generator_outputs(self):
        """Abstract function for decoding the Generators' outputs through the Encoders.

        Returns:
            cond_probs: Flattened conditional probabilities for the decoded outputs.
            decoded_outputs: Flattened decoded generator outputs.
        """
        pass

    def _build_variables(self):
        """Collects MultINN model variables."""
        self._variables = {
            'encoders': [e.variables for e in self.encoders],
            'generators': [g.variables for g in self.generators],
            'feedback': self.trainable_feedback_variables
        }

        self._trainable_variables = \
            self.trainable_encoder_variables \
            + self.trainable_generator_variables \
            + self.trainable_feedback_variables

    def sampler(self, num_beats):
        """Returns the sequence sampler op, the outputs from `model.generate()`.

        Args:
            num_beats: A number of beats to sample.

        Returns:
            samples: A batch of generated sequences,
                sized `[batch_size, num_steps, num_dims]`
        """
        data_config = self._config['data']
        beat_resolution = data_config['beat_resolution']
        pitch_span = data_config['pitch_range']['highest'] - data_config['pitch_range']['lowest']

        with tf.name_scope(f'sampler/'):
            # Computing the number of time steps in input dimensions.
            num_steps = num_beats * beat_resolution * pitch_span // self.num_dims
            return self.generate(num_steps)

    def evaluator(self):
        """Returns the sequence musical metrics evaluation ops.

        Returns:
            samples: A batch of generated sequences,
                sized `[batch_size, num_steps, num_dims]`
        """
        beat_resolution = self._config['data']['beat_resolution']
        pitch_span = self._config['data']['pitch_range']['highest'] - self._config['data']['pitch_range']['lowest']

        with tf.name_scope(f'evaluator/'):
            # Reshaping the inputs into bar-music
            #   from [batch, time_steps, num_dims, num_tracks]
            #     to [batch, bars, bar_steps, pitch_span, num_tracks]
            bar_music = tf.reshape(
                self._x,
                [tf.shape(self._x)[0], -1, 4 * beat_resolution, pitch_span, self.num_tracks]
            )

            return get_metric_summary_ops(bar_music, self.tracks)

    def _combine_track_metrics(self, track_metrics, track_metrics_upd, track_summaries, global_scope=None):
        """Auxiliary function for combining metrics from multiple tracks.

        Args:
            track_metrics: A list of metrics dictionaries for each track.
            track_metrics_upd: A list of metrics update ops for each track.
            track_summaries: A list of metrics summary dictionaries for each track.
            global_scope: Optional global scope name for computing average metrics
                on a global multi-track level.

        Returns:
            metrics: A combined dictionary of metric names and metric ops.
            metrics_upd: A combined list of metric update ops.
            metric_summaries: A merged tf.Summary object for model's metrics.
        """
        assert len(track_metrics) == self.num_tracks

        metrics = {}
        metrics_upd = []
        summaries = {}

        # Collecting metrics and summaries from each track
        for i in range(self.num_tracks):
            for k, m in track_metrics[i].items():
                if k in metrics:
                    metrics[k].append(m)
                else:
                    metrics[k] = [m]

            metrics_upd += track_metrics_upd[i]

            for k, s in track_summaries[i].items():
                if k in summaries:
                    summaries[k].append(s)
                else:
                    summaries[k] = [s]

        # Computing global track-level metrics
        if global_scope is not None:
            with tf.variable_scope(global_scope):
                for k in metrics.keys():
                    metrics[k] = tf.reduce_mean(metrics[k], name=k)

                summary_global = [tf.summary.scalar(k, m) for k, m in metrics.items() if k != 'batch/loss']
                summaries['metrics'].append(tf.summary.merge(summary_global))

        for k, s in summaries.items():
            summaries[k] = tf.summary.merge(s)

        return metrics, metrics_upd, summaries

    @property
    def trainable_encoder_variables(self):
        """A list of MultINN's trainable Encoder variables."""
        return list(itertools.chain.from_iterable([e.trainable_variables for e in self.encoders]))

    @property
    def trainable_generator_variables(self):
        """A list of MultINN's trainable Generator variables."""
        return list(itertools.chain.from_iterable([g.trainable_variables for g in self.generators]))

    def load_encoders(self, sess, ckpt_dir):
        """Restores MultINN Encoders' state from the checkpoint directory.

        Args:
            sess: A tf.Session object to use to restore variables.
            ckpt_dir: The checkpoint directory where the weights
                should be restored from.

        Returns:
            success: `bool` indicating if Encoders' weights were restored
                successfully or not.
        """
        if self.encoder_type != 'Pass':
            saver = tf.train.Saver(self.trainable_encoder_variables)

            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(sess, os.path.join(ckpt_dir, ckpt_name))
                return True
            else:
                return False
        else:
            return True
