"""Implementation of an interface for the MultINN framework."""

import abc

from models.common.model import Model


class MultINNInterface(Model):
    """MultINN framework interface.

    Defines the main properties and functions of MultINN models.
    """

    def __init__(self, name='MultINN'):
        """Initializes MultINN model as a Model object.

        Args:
            name: Optional model name to use as a prefix when adding operations.
        """
        super().__init__(name=name)

        pass

    @property
    @abc.abstractmethod
    def mode(self):
        """str: MultINN operation mode."""
        pass

    @property
    @abc.abstractmethod
    def num_dims(self):
        """int: The number of input dimensions the models."""
        pass

    @property
    @abc.abstractmethod
    def tracks(self):
        """list of str: The names of tracks with which MultINN operates."""
        pass

    @property
    @abc.abstractmethod
    def num_tracks(self):
        """int: The number of tracks with which MultINN operate."""
        pass

    @property
    @abc.abstractmethod
    def encoder_type(self):
        """str: The model type of Encoders that MultINN utilizes."""
        pass

    @property
    @abc.abstractmethod
    def generator_type(self):
        """str: The model type of Generators that MultINN utilizes."""
        pass

    @property
    @abc.abstractmethod
    def encoders(self):
        """The list of the MultINN Encoders."""
        pass

    @property
    @abc.abstractmethod
    def generators(self):
        """The list of the MultINN Generators."""
        pass

    @property
    @abc.abstractmethod
    def feedback_module(self):
        """`bool` indicating whether the model utilized feedback modules."""
        pass

    @property
    @abc.abstractmethod
    def keep_prob(self):
        """float: Dropout keep probability."""
        pass

    @property
    @abc.abstractmethod
    def tune_encoder(self):
        """`bool` indicating whether to update Encoders' parameters while training generators."""
        pass

    @property
    @abc.abstractmethod
    def loss(self):
        """MultINN model loss op."""
        pass

    @abc.abstractmethod
    def generate(self, num_steps):
        """Abstract function for generating new sequences with the MultINN.

        The generative process starts by encoding the batch of input sequences
        and going through them to obtain the final inputs' state.
        Then, the Generators generates`num_steps` new steps by sampling
        from the modelled conditional distributions.
        Finally, samples are decoded through the Encoders to get the
        samples in original multi-track input format.

        Each MultINN model implementation should define the function explicitly.

        Args:
            num_steps: A number of time steps to generate.

        Returns:
            samples: A batch of generated sequences,
                sized `[batch_size, num_steps, num_dims, num_tracks]`
        """
        pass

    @abc.abstractmethod
    def sampler(self, num_beats):
        """Abstract function for returning the sequence sampler op.

        Each MultINN model implementation should define the function explicitly.
        The function should return the outputs from `model.generate()`.

        Args:
            num_beats: A number of beats to sample.

        Returns:
            samples: A batch of generated sequences,
                sized `[batch_size, num_steps, num_dims]`
        """
        pass

    @abc.abstractmethod
    def evaluator(self):
        """Abstract function for returning the sequence evaluation ops.

        Each MultINN model implementation should define the function explicitly.
        The function should return the ops for evaluation the inputs
        using musical metrics.

        Returns:
            samples: A batch of generated sequences,
                sized `[batch_size, num_steps, num_dims]`
        """
        pass

    @abc.abstractmethod
    def train_encoders(self, optimizer, lr, layer=0):
        """Constructs ops for training MultINN Encoders.

        Each MultINN model implementation should define the function explicitly.

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
        pass

    @abc.abstractmethod
    def pretrain_generators(self, optimizer, lr, separate_losses=False):
        """Constructs training ops for pre-training MultINN Generators.

        Each MultINN model implementation should define the function explicitly.

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
        pass

    @abc.abstractmethod
    def train_generators(self, optimizer, lr, separate_losses=False):
        """Constructs training ops for training MultINN Generators.

        Each MultINN model implementation should define the function explicitly.

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
        pass

    @property
    @abc.abstractmethod
    def trainable_encoder_variables(self):
        """A list of MultINN's trainable Encoder variables."""
        pass

    @property
    @abc.abstractmethod
    def trainable_generator_variables(self):
        """A list of MultINN's trainable Generator variables."""
        pass

    @property
    @abc.abstractmethod
    def trainable_feedback_variables(self):
        """A list of MultINN's trainable feedback module variables."""
        pass

    @abc.abstractmethod
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
        pass
