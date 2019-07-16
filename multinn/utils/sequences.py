"""Functions for working with sequences."""

import tensorflow as tf


def flatten_maybe_padded_sequences(tensor, lengths=None):
    """Flattens the batch of sequences, removing padding (if applicable).

    Note:
        Taken from: https://github.com/tensorflow/magenta/blob/master/magenta/common/sequence_example_lib.py

    Args:
        tensor: A tensor of possibly padded sequences to flatten,
            sized `[N, M, ...]` where M = max(lengths).
        lengths: Optional length of each sequence, sized `[N]`. If None, assumes no padding.

    Returns:
        flatten_maybe_padded_sequences: The flattened sequence tensor, sized
            `[sum(lengths), ...]`.
    """

    def flatten_unpadded_sequences():
        # The sequences are equal length, so we should just flatten over the first two dimensions.
        return tf.reshape(tensor, tf.concat([[-1], tf.shape(tensor)[2:]], 0))

    if lengths is None:
        return flatten_unpadded_sequences()

    def flatten_padded_sequences():
        indices = tf.where(tf.sequence_mask(lengths))
        return tf.gather_nd(tensor, indices)

    return tf.cond(
        tf.equal(tf.reduce_min(lengths), tf.shape(tensor)[1]),
        flatten_unpadded_sequences,
        flatten_padded_sequences
    )


def flatten_sequences_and_save_shape(tensor):
    """Flattens the batch sequences and returns the original batch-time shape.

    Args:
        tensor: A tensor of sequences to flatten,
            sized `[N, M, ...]`, or `[N, ...]`.

    Returns:
        tensor: The flattened tensor,
            sized `[NxM, ...]` or `[N, ...]`.
        batch_time_shape: The batch-time shape,
            either `(N, M,)` or `(N,)`.
    """
    if tensor.shape.ndims > 2:
        batch_time_shape = tf.shape(tensor)[:2]
        tensor = tf.reshape(tensor, tf.concat([[-1], tf.shape(tensor)[2:]], 0))
    else:
        batch_time_shape = tf.shape(tensor)[:1]

    return tensor, batch_time_shape


def reshape_to_batch_time(tensor, batch_time_shape):
    """Reshapes the tensor into batch-time shape.

    Args:
        tensor: The flattened tensor,
            sized `[NxM, ...]` or `[N, ...]`.
        batch_time_shape: The batch-time shape,
            either `(N, M,)` or `(N,)`.

    Returns:
        batch_time_tensor: The tensor in batch-time shape,
            sized `[N, M, ...]` or `[N, ...]`.
    """
    return tf.reshape(
        tensor,
        tf.concat([batch_time_shape, tf.shape(tensor)[1:]], 0)
    )
