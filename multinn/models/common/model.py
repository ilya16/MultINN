"""This module contains the base class for all models utilized in the MultINN framework."""

import abc
import os

import tensorflow as tf


class Model:
    """Abstract class for Neural Network models.

    Defines the main attributes of models: TensorFlow variables, placeholders,
    metrics, metrics update ops, and summaries.

    Defines the interface for model graph construction (model.build()).

    Defines the model save and restore methods.

    Note:
        model.build() should be called after initializing and before using any model,
        until the method is called inside object construction.
    """

    def __init__(self,
                 name=None):
        """Initializes the base parameters and variables of the model.

        Args:
            name: Optional model name to use as a prefix when adding operations.
        """
        self._name = name

        # Model variables
        self._placeholders = {}
        self._variables = {}
        self._trainable_variables = []

        # Attributes assigned during the graph construction (model.build())
        self._inputs = None
        self._outputs = None
        self._is_train = None
        self._saver = None

        self._is_built = False

        # Monitoring variables assigned during the graph construction
        self._metrics = {}
        self._metrics_upd = []
        self._summaries = {}

    @property
    def name(self):
        """str: The name of the model."""
        return self._name

    @property
    def is_built(self):
        """bool: Is model built or not."""
        return self._is_built

    @property
    def variables(self):
        """A dictionary of model's variables grouped by modules."""
        return self._variables

    @property
    def trainable_variables(self):
        """A list of model's trainable variables."""
        return self._trainable_variables

    @property
    def placeholders(self):
        """A dictionary of model's placeholders.

        Placeholders to use during the model training and inference.
        """
        return self._placeholders

    @property
    def metrics(self):
        """A dictionary of model's metrics.

        Metrics to optimize and monitor during the model training and inference.
        """
        return self._metrics

    @property
    def metrics_upd(self):
        """A list of metric update ops."""
        return self._metrics_upd

    @property
    def summaries(self):
        """A dictionary of summaries to monitor in TensorBoard.

        Note:
            Dictionary of summaries should follow the convection:
                'metrics': a merged summary of model's metrics.
                'weights': a merged summary of variables' weights.
                'gradients': a merged summary of gradients.
        """
        return self._summaries

    @property
    def weight_summary(self):
        """tf.Summary: model weights summary for all trainable model variables."""
        return tf.summary.merge([tf.summary.histogram(var.name, var) for var in self.trainable_variables])

    @property
    def saver(self):
        """tf.Saver: model's saver object."""
        return self._saver

    @abc.abstractmethod
    def build(self,
              x=None,
              y=None,
              lengths=None,
              is_train=None,
              mode='eval'):
        """Abstract function for building the model's graph.

        Each model should define the function explicitly.

        The function should assign the model's variables, trainable variables,
        placeholders, metrics, and summaries unless these attributes are assigned
        during the model initialization or these attributes are not needed by
        the model's design.

        Each model should assign `self._is_built = True` to assure that
        model is built and can be used for training or inference.

        Args:
            x: Optional inputs to the model,
                sized `[batch_size, time_steps, num_input_dims]`.
            y: Optional targets for the model,
                sized `[batch_size, time_steps, num_output_dims]`.
            lengths: Optional lengths of input sequences sized `[batch_size]`.
            is_train: Optional `bool` for distinguishing between training and inference.
            mode: Build mode for optimizing the graph size.
                Some operations may not be needed for training or inference.

        Raises:
            ValueError: If incorrect build mode is provided.
        """
        if mode not in ('train', 'eval', 'generate'):
            raise ValueError("Incorrect model build `mode`."
                             "Possible options are 'train', 'eval', or 'generate'. "
                             "The `mode` parameter was: %s" % mode)

    @abc.abstractmethod
    def build_metrics(self,
                      targets,
                      predictions,
                      cond_probs=None,
                      log_probs=None):
        """Abstract function for building the model's metrics.

        Each model should define the function explicitly.

        Models can either have or do not have evaluation metrics.

        Args:
            targets: Flattened target predictions,
                sized `[batch_size, num_output_dims]`.
            predictions: Flattened model's predictions,
                sized `[batch_size, num_output_dims]`.
            cond_probs: Flattened conditional probabilities for predictions,
                sized `[batch_size, num_output_dims]`.
            log_probs: Flattened log probabilities for predictions,
                sized `[batch_size]`.

        Returns:
            metrics: A dictionary of metric names and metric ops.
            metrics_upd: A list of metric update ops.
            metric_summaries: A tf.Summary object for model's metrics.
        """
        pass

    def save(self,
             sess,
             ckpt_dir,
             global_step=None,
             write_meta_graph=False):
        """Saves model's state at the checkpoint directory.

        Args:
            sess: A tf.Session object to use to save variables.
            ckpt_dir: The checkpoint directory where the weights should be saved to.
            global_step: Optional global step number to be appended
                to the checkpoint filenames.
            write_meta_graph: `bool` indicating whether or not to write the
                meta graph file.
        """
        # Initializing the Saver object on the first call.
        if self.saver is None:
            self._saver = tf.train.Saver(self.trainable_variables)

        model_name = self._name + '.model'

        # Assuring the checkpoint directory exists.
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        self.saver.save(sess,
                        os.path.join(ckpt_dir, model_name),
                        global_step=global_step,
                        write_meta_graph=write_meta_graph)

    def load(self,
             sess,
             ckpt_dir):
        """Restores model's state from the checkpoint directory.

        Args:
            sess: A tf.Session object to use to restore variables.
            ckpt_dir: The checkpoint directory where the weights
                should be restored from.

        Returns:
            success: `bool` indicating if model's weights were restored
                successfully or not.
        """
        # Initializing the Saver object on the first call.
        if self.saver is None:
            self._saver = tf.train.Saver(self.trainable_variables)

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(ckpt_dir, ckpt_name))
            return True
        else:
            return False
