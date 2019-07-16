"""Implementation of an interface for Generator modules in the MultINN framework."""

import abc

from models.common.model import Model
from utils.training import compute_gradients


class Generator(Model):
    """Abstract sequence Generator."""

    def __init__(self,
                 num_dims,
                 num_hidden,
                 num_hidden_rnn,
                 keep_prob=1.0,
                 internal_bias=False,
                 name='generator',
                 track_name='all'):
        """Initializes the base parameters and variables of the Generator.

        Args:
            num_dims: The number of output dimensions of the Generator.
            num_hidden: The number of hidden units of the Generator.
            num_hidden_rnn: The number of hidden units of the temporal RNN cell.
            keep_prob: The portion of outputs to use during the training,
                inverse of `dropout_rate`.
            internal_bias: `bool` indicating whether the Generator should maintain
                its own bias variables. If `False`, external values must be passed.
            name: The model name to use as a prefix when adding operations.
            track_name: The track name on which encoder operates.
        """
        super().__init__(name=name)
        self._track_name = track_name

        if isinstance(num_hidden, int):
            num_hidden = [num_hidden]

        if isinstance(num_hidden_rnn, int):
            num_hidden_rnn = [num_hidden_rnn]

        self._num_dims = num_dims
        self._num_hidden = num_hidden
        self._num_hidden_rnn = num_hidden_rnn

        self._keep_prob = keep_prob
        self._internal_bias = internal_bias

        self._lengths = None

    @property
    def num_dims(self):
        """int: The number of output dimensions of the Generator."""
        return self._num_dims

    @property
    def num_hidden(self):
        """list of int: The number of hidden units for all layers of the Generator."""
        return self._num_hidden

    @property
    def num_hidden_rnn(self):
        """list of int: The number of hidden units for all layers of the RNN cell."""
        return self._num_hidden_rnn

    @property
    def track_name(self):
        """str: The name of the track on which the Encoder operates."""
        return self._track_name

    @property
    def keep_prob(self):
        """float: Dropout keep probability."""
        return self._keep_prob

    @property
    def internal_bias(self):
        """bool: The number of hidden units for each input/output of the NADE."""
        return self._internal_bias

    @abc.abstractmethod
    def build(self, x=None, y=None, lengths=None, is_train=None, mode='eval'):
        """Abstract function for building the Generator's graph.

        Each generator implementation should define the function explicitly.

        The function should assign the model's variables, trainable variables,
        placeholders, metrics, and summaries unless these attributes are assigned
        during the model initialization or these attributes are not needed by
        the encoder's design.

        Each model should assign `self._is_built = True` to assure that
        generator is built and can be used for training or inference.

        Args:
            x: A batch of inputs to the generator,
                sized `[batch_size, time_steps, ...]`.
            y: A batch of target predictions for the generator,
                sized `[batch_size, time_steps, num_dims]`.
            lengths: Lengths of input sequences sized `[batch_size]`.
            is_train: `bool` for distinguishing between training and inference.
            mode: Build mode for optimizing the graph size.
                Some operations may not be needed for training or inference.

        Raises:
            ValueError: If incorrect build mode is provided.
        """
        super().build(mode=mode)

        self._inputs = x
        self._lengths = lengths

    @abc.abstractmethod
    def zero_state(self, batch_size):
        """Returns the initial state of the Generator.

        Each generator implementation should define the function explicitly.

        Args:
            batch_size: The batch size of inputs to the cell.

        Returns:
            initial_state: Generator's initial state.
        """
        pass

    def forward(self):
        """Returns Generator outputs from the sequential forward pass."""
        return self._outputs

    @abc.abstractmethod
    def generate(self, x, num_steps):
        """Abstract function for generating new sequences with the Generator.

        The generative process starts by going through the batch of
        input sequences `x` and obtaining the final inputs' state.
        Then, the generator generates `num_steps` new steps by sampling
        from the modelled conditional distributions.

        Args:
            x: A batch of inputs to the generator,
                sized `[batch_size, time_steps, ...]`.
            num_steps: A number of time steps to generate.

        Returns:
            samples: A batch of generated sequences,
                sized `[batch_size, num_steps, num_dims]`
        """
        pass

    @abc.abstractmethod
    def pretrain(self, optimizer, lr, run_optimizer=True):
        """Constructs training ops for pre-training the Generator.

        Some generator implementation may offer the pre-training of
        some of the generator modules and should define this function explicitly.

        Args:
            optimizer: tf.Optimizer object that computes gradients.
            lr: A learning rate for weight updates.
            run_optimizer: `bool` indicating whether to run optimizer
                on the generator's trainable variables, or return
                an empty update ops list. Allows running the optimizer
                on the top of multiple generators.

        Returns:
            init_ops: A list of weight initialization ops.
            update_ops: A list of weights update ops.
            metrics: A dictionary of metric names and metric ops.
            metrics_upd: A list of metric update ops.
            summaries: A dict of tf.Summary objects for model's
                metrics, weights and training gradients.
        """
        pass

    def train(self, optimizer, lr, run_optimizer=True):
        """Constructs training ops for the Generator.

        Args:
            optimizer: tf.Optimizer object that computes gradients.
            lr: A learning rate for weight updates.
            run_optimizer: `bool` indicating whether to run optimizer
                on the generator's trainable variables, or return
                an empty update ops list. Allows running the optimizer
                on the top of multiple generators.

        Returns:
            init_ops: A list of weight initialization ops.
            update_ops: A list of weights update ops.
            metrics: A dictionary of metric names and metric ops.
            metrics_upd: A list of metric update ops.
            summaries: A dict of tf.Summary objects for model's
                metrics, weights and training gradients.
        """
        # updates for the whole RNN-RBM
        loss = self.metrics['batch/loss']
        summaries = self.summaries.copy()

        init_ops = []
        if run_optimizer:
            update_ops, summaries['gradients'] = compute_gradients(optimizer, loss, self.trainable_variables)
        else:
            update_ops = []

        return init_ops, update_ops, self.metrics, self.metrics_upd, summaries
