from models.multinn.core.multinn_interface import MultINNInterface
from models.multinn.multinn_composer import MultINNComposer
from models.multinn.multinn_feedback import MultINNFeedback
from models.multinn.multinn_feedback_rnn import MultINNFeedbackRnn
from models.multinn.multinn_jamming import MultINNJamming
from models.multinn.multinn_joint import MultINNJoint


class MultINN(MultINNInterface):
    """MultINN model.

    MultINN – Multi-Instrumental Neural Network for multi-track
    sequence generation with inter-track relations and multi-modal input distributions.

    The Python class acts as a wrapper around concrete MultINN implementation.
    Supported models are:
        Joint MultiNN – MultiNNJoint
        Composer MultiNN – MultiNNComposer
        Jamming MultiNN – MultiNNJamming
        Feedback MultiNN – MultINNFeedback
        Feedback-RNN MultiNN – MultINNFeedbackRnn
    """

    def __init__(self, config, params, mode='feedback-rnn', name='MultINN'):
        """Initializes MultINN model based on the operation mode.

        Args:
            config: A dict of the experiment configurations.
            params: A dict of the model parameters.
            mode: A MultINN model type, operation mode.
                Supported types: `joint`, `composer`, `jamming`,
                 `feedback`, and `feedback-rnn`.
            name: Optional model name to use as a prefix when adding operations.
        """
        super().__init__(name=name)

        if mode == 'joint':
            self._model = MultINNJoint(config, params, name=name)
        elif mode == 'composer':
            self._model = MultINNComposer(config, params, name=name)
        elif mode == 'jamming':
            self._model = MultINNJamming(config, params, name=name)
        elif mode == 'feedback':
            self._model = MultINNFeedback(config, params, name=name)
        elif mode == 'feedback-rnn':
            self._model = MultINNFeedbackRnn(config, params, name=name)
        else:
            raise ValueError('Incorrect operation mode, choose from `joint`, '
                             '`composer`, `jamming`, `feedback`, and `feddback-rnn`.')

        self._placeholders = self._model.placeholders

        self.build()

    @property
    def mode(self):
        """str: MultINN operation mode."""
        return self._model.mode

    @property
    def num_dims(self):
        """int: The number of input dimensions the models."""
        return self._model.num_dims

    @property
    def tracks(self):
        """list of str: The names of tracks with which MultINN operates."""
        return self._model.tracks

    @property
    def num_tracks(self):
        """int: The number of tracks with which MultINN operate."""
        return self._model.num_tracks

    @property
    def encoder_type(self):
        """str: The model type of Encoders that MultINN utilizes."""
        return self._model.encoder_type

    @property
    def generator_type(self):
        """str: The model type of Generators that MultINN utilizes."""
        return self._model.generator_type

    @property
    def encoders(self):
        """The list of the MultINN Encoders."""
        return self._model.encoders

    @property
    def generators(self):
        """The list of the MultINN Generators."""
        return self._model.generators

    @property
    def feedback_module(self):
        """`bool` indicating whether the model utilized feedback modules."""
        return self._model.feedback_module

    @property
    def keep_prob(self):
        """float: Dropout keep probability."""
        return self._model.keep_prob

    @property
    def tune_encoder(self):
        """`bool` indicating whether to update encoders' parameters while training generators."""
        return self._model.tune_encoder

    @property
    def loss(self):
        """MultINN model loss op."""
        return self._model.loss

    def build(self, x=None, y=None, lengths=None, is_train=None, mode='eval'):
        """Builds MultINN model's graph.

        Concretely, build internal MultINN model implementation.

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

        self._model.build(x, y, lengths, is_train, mode)

        self._metrics = self._model.metrics
        self._metrics_upd = self._model.metrics_upd
        self._summaries = self._model.summaries

        self._variables = self._model.variables
        self._trainable_variables = self._model.trainable_variables

        self._is_built = True

    def build_metrics(self, targets, predictions, cond_probs=None, log_probs=None):
        """Builds MultINN model's metrics using internal MultINN implementation.

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
        return self._model.build_metrics(targets, predictions, cond_probs, log_probs)

    def generate(self, num_steps):
        """Generates new sequences through the internal MultINN implementation.

        The generative process starts by encoding the batch of input sequences
        and going through them to obtain the final inputs' state.
        Then, the Generators generates`num_steps` new steps by sampling
        from the modelled conditional distributions.
        Finally, samples are decoded through the Encoders to get the
        samples in original multi-track input format.

        Args:
            num_steps: A number of time steps to generate.

        Returns:
            samples: A batch of generated sequences,
                sized `[batch_size, num_steps, num_dims, num_tracks]`
        """
        return self._model.generate(num_steps)

    def sampler(self, num_beats):
        """Returns the sequence sampler op, the outputs from `model.generate()`.

        Args:
            num_beats: A number of beats to sample.

        Returns:
            samples: A batch of generated sequences,
                sized `[batch_size, num_steps, num_dims]`
        """
        return self._model.sampler(num_beats)

    def evaluator(self):
        """Returns the sequence musical metrics evaluation ops.

        Returns:
            samples: A batch of generated sequences,
                sized `[batch_size, num_steps, num_dims]`
        """
        return self._model.evaluator()

    def train_encoders(self, optimizer, lr, layer=0):
        """Constructs ops for training MultINN Encoders.

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
        return self._model.train_encoders(optimizer, lr, layer=layer)

    def pretrain_generators(self, optimizer, lr, separate_losses=False):
        """Constructs training ops for pre-training MultINN Generators.

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
        return self._model.pretrain_generators(optimizer, lr)

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
        return self._model.train_generators(optimizer, lr)

    @property
    def trainable_encoder_variables(self):
        """A list of MultINN's trainable Encoder variables."""
        return self._model.trainable_encoder_variables

    @property
    def trainable_generator_variables(self):
        """A list of MultINN's trainable Generator variables."""
        return self._model.trainable_generator_variables

    @property
    def trainable_feedback_variables(self):
        """A list of MultINN's trainable feedback module variables."""
        return self._model.trainable_feedback_variables

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
        return self._model.load_encoders(sess, ckpt_dir)
