"""Implementation of an interface for Encoder modules in the MultINN framework."""

import abc

from models.common.model import Model


class Encoder(Model):
    """Abstract multi-layer Encoder."""

    def __init__(self,
                 num_dims,
                 num_hidden,
                 name='encoder',
                 track_name='all'):
        """Initializes the base parameters and variables of the Encoder.

        Args:
            num_dims: The number of input dimensions of the Encoder.
            num_hidden: The number of hidden units of the Encoder.
            name: The model name to use as a prefix when adding operations.
            track_name: The track name on which encoder operates.
        """
        super().__init__(name=name)
        self._track_name = track_name

        if isinstance(num_hidden, int):
            num_hidden = [num_hidden]

        self._num_dims = num_dims
        self._num_hidden = num_hidden

        # Additional attributes assigned during the graph construction
        self._lengths = None
        self._enc_probs, self._encodings = [], []
        self._dec_probs, self._decodings = [], []

    @property
    def num_dims(self):
        """int: The number of input/output dimensions of the Encoder."""
        return self._num_dims

    @property
    def num_hidden(self):
        """list of int: The number of hidden units for all layers of the Encoder."""
        return self._num_hidden

    @property
    def num_layers(self):
        """int: The number of layers in the Encoder."""
        return len(self._num_hidden)

    @property
    def track_name(self):
        """str: The name of the track on which the Encoder operates."""
        return self._track_name

    @property
    def encodings(self):
        """A list of encodings for each layer of the Encoder."""
        return self._encodings

    @property
    def decodings(self):
        """A list of decodings for each layer of the Encoder."""
        return self._decodings

    @property
    def enc_probs(self):
        """A list of conditional probabilities for encodings for each layer of the Encoder."""
        return self._enc_probs

    @property
    def dec_probs(self):
        """A list of conditional probabilities for decodings for each layer of the Encoder."""
        return self._dec_probs

    @abc.abstractmethod
    def build(self, x=None, y=None, lengths=None, is_train=None, mode='eval'):
        """Abstract function for building the Encoder's graph.

        Each encoder implementation should define the function explicitly.

        The function should assign the model's variables, trainable variables,
        placeholders, metrics, and summaries unless these attributes are assigned
        during the model initialization or these attributes are not needed by
        the encoder's design.

        Each model should assign `self._is_built = True` to assure that
        encoder is built and can be used for training or inference.

        Args:
            x: Optional inputs to the encoder,
                sized `[batch_size, time_steps, num_dims]`.
            y: Optional targets for the encoder,
                sized `[batch_size, time_steps, num_hidden[-1]]`.
            lengths: Optional lengths of input sequences sized `[batch_size]`.
            is_train: Optional `bool` for distinguishing between training and inference.
            mode: Build mode for optimizing the graph size.
                Some operations may not be needed for training or inference.

        Raises:
            ValueError: If incorrect build mode is provided.
        """
        super().build(x, y, lengths, is_train, mode)

        self._inputs = x
        self._lengths = lengths

    @abc.abstractmethod
    def encode(self, x=None):
        """Encodes inputs through all layers.

        Each encoder implementation should define the function explicitly.

        Args:
            x: A batch of inputs to the Encoder,
                sized `[batch_size, time_steps, num_dims]` or `[batch_size, num_dims]`,
                or None if `self.encodings` from the graph construction should be used.

        Returns:
            p_h: The conditional probabilities for encodings on the last hidden layer,
                sized `[batch_size, num_hidden[-1]]`.
            h: A batch of encoded outputs,
                sized `[batch_size, num_hidden[-1]]`.
        """
        pass

    @abc.abstractmethod
    def decode(self, h=None):
        """Decodes hidden vectors on the last layer through all layers.

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
        pass

    @abc.abstractmethod
    def train(self, optimizer, lr, layer=None):
        """Constructs training ops for the Encoder.

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
