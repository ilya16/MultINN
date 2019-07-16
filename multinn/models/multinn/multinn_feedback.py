"""Implementation of a Feedback MultINN model with per-track modules and feedback module."""

import tensorflow as tf

from models.common.dnn import DNN
from models.multinn.multinn_jamming import MultINNJamming
from utils.auxiliary import get_current_scope


class MultINNFeedback(MultINNJamming):
    """Feedback MultINN model.

    Multiple per-track Encoders + Multiple per-track Generators
    + Feedback module on the Generators layer.

    Works with multi-track sequences and learns the track specific features.
    The Feedback layer offers inter-track relation modelling.
    The Generators work with own track sequences and receive
    the processed feedback about the whole multi-track sequence.

    The Feedback module is a Dense NN that processes only one step.
    The Feedback module can be substituted with over Neural Networks
    (see Feedback-RNN MultINN with RNN for the Feedback layer).
    """

    def __init__(self, config, params, name='MultINN-feedback'):
        """Initializes Feedback MultiNN model.

        Args:
            config: A dict of the experiment configurations.
            params: A dict of the model parameters.
            name: Optional model name to use as a prefix when adding operations.
        """
        super().__init__(config, params, name=name)
        self._mode = 'feedback'

        self._feedback_module = True
        self._init_feedback()

        self._x_encoded_stack = None
        self._x_feedback = None
        self._feedback_final_state = None

        self._trainable_feedback_variables = []

    def _init_feedback(self):
        """Initializes Dense Feedback module of the Feedback MultINN model."""
        self._feedback_layer = DNN(
            num_units=self._params['generator']['feedback'],
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        )

    def _build_generators(self, mode='eval'):
        """Building the Feedback MultINN Generators and the Feedback module.

        Inputs to the Generators are outputs from the Encoders,
        concatenated with the stacked encoded inputs passed through
        the Feedback layer (module).

        Feedback module can be a Dense NN (Feedback MultINN),
        or a Recurrent NN (Feedback-RNN MultINN).

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

        # Building the Feedback module
        with tf.variable_scope(f'{get_current_scope()}feedback/'):
            self._feedback_layer.build(is_train=self._is_train)

            self._x_feedback, self._feedback_final_state = self._apply_feedback(
                self._x_encoded_stack,
                single_step=False
            )

        # Building the per-track Generators
        for i in range(self.num_tracks):
            with tf.variable_scope(f'generator_inputs/{self.tracks[i]}'):
                inputs = tf.concat([self._x_encoded[i], self._x_feedback], axis=-1)
                generator_inputs = inputs[:, 0:-1, :]

            with tf.variable_scope(f'generator_targets/{self.tracks[i]}'):
                generator_targets = self._x_encoded[i][:, 1:, :]

            self.generators[i].build(x=generator_inputs, y=generator_targets,
                                     lengths=self._lengths, is_train=self._is_train, mode=mode)

        # Saving trainable Feedback module variable
        self._trainable_feedback_variables = self._feedback_layer.trainable_variables

    def _apply_feedback(self, inputs, lengths=None, initial_state=None, single_step=False):
        """Runs the Dense NN Feedback module over the stacked encoded inputs.

        Args:
            inputs: A batch of stacked encoded inputs,
                sized `[batch_size, time_steps, num_tracks x num_dims]`,
                or    `[batch_size, num_tracks x num_dims]`.
            lengths: The lengths of input sequences sized `[batch_size]`,
                or None if all are equal.
            initial_state: The initial state of the Feedback module,
                or None if the zero state should be used
                or there is no Feedback module state.
            single_step: `bool` indicating whether to run the Feedback module
                over a single step sequence or not.

        Returns:
             feedback_outputs: A batch of feedback vectors,
                sized [batch_size, time_steps, num_feedback].
        """
        return self._feedback_layer(inputs), tf.zeros(1)

    def generate(self, num_steps):
        """Generating new sequences with the Feedback MultINN.

        The generative process starts by encoding the batch of input sequences
        and going through them to obtain the final inputs' state.
        Then, the per-track Generators generate `num_steps` new steps
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
        with tf.variable_scope('batch_size'):
            batch_size = tf.shape(self._inputs)[0]

        intro_states = []
        # Generators' forward pass
        for i in range(self.num_tracks):
            with tf.variable_scope(f'inputs/{self.tracks[i]}'):
                inputs = tf.concat([self._x_encoded[i], self._x_feedback], axis=-1)

            with tf.variable_scope(f'intro_state/{self.tracks[i]}'):
                state = self.generators[i].steps(inputs)
                intro_states.append(state)

        #
        with tf.variable_scope('feedback_sampler'):
            samples_h, _, _ = tf.scan(
                self._feedback_recurrence,
                tf.zeros((num_steps, 1)),
                initializer=(
                    tf.zeros((batch_size, self._num_dims_generator, self.num_tracks)),
                    intro_states,
                    self._feedback_final_state)
            )

        with tf.variable_scope('samples/encoded/'):
            samples_h = tf.unstack(tf.transpose(samples_h, [1, 0, 2, 3]), axis=-1)

        for i in range(self.num_tracks):
            # Decoding inputs into the original format
            with tf.variable_scope(f'samples/{self.tracks[i]}/'):
                _, samples = self.encoders[i].decode(samples_h[i])

                music.append(samples)

        with tf.variable_scope('samples/'):
            return tf.stack(music, axis=3, name='music')

    def _feedback_recurrence(self, val, _):
        """Auxiliary function for recursive sampling.

        Samples from the passed state and computes the next Generators' and Feedback states.

        Args:
            val: A tuple (intro, states, feedback_state) of samples from the last time step
                and the current Generators' and Feedback module states.

        Returns:
            samples: The samples for the current state,
                sized `[batch_size, num_dims, num_tracks]`
            generator_states: The new Generator states.
            feedback_state: The new Feedback module state.
        """
        # Unpacking previous step data
        prev_samples, states, feedback_state = val

        # Sampling values with each Generator
        samples_list = []
        for i in range(self.num_tracks):
            samples_i, _ = self.generators[i].sample_single(prev_samples, states[i])
            samples_list.append(samples_i)

        samples = tf.stack(samples_list, axis=-1)
        batch_size = tf.shape(samples)[0]

        # Stacking inputs and running them through the Feedback module
        samples_stack = tf.reshape(samples, [batch_size, self._num_dims_generator * self.num_tracks])
        x_feedback, feedback_state = self._apply_feedback(
            samples_stack,
            initial_state=feedback_state,
            single_step=True
        )

        # Computing new Generator states
        generator_states = []
        for i in range(self.num_tracks):
            inputs_i = tf.concat([samples_list[i], x_feedback], axis=1)

            state_i_new = self.generators[i].single_step(inputs_i, states[i])
            generator_states.append(state_i_new)

        return samples, generator_states, feedback_state

    @property
    def trainable_feedback_variables(self):
        """An list of the Feedback MultINN trainable feedback module variables."""
        return self._trainable_feedback_variables
