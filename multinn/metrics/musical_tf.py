"""Implementation of the musical metrics for evaluating the music samples.

The metrics are implemented using TensorFlow.

Note:
    Adapted from: https://github.com/salu133445/musegan/blob/master/src/musegan/metrics.py
"""

import tensorflow as tf

from metrics.musical import qualified_note_rate, harmonicity, drum_in_pattern_rate


# Utility functions
def tf_to_chroma(pianoroll):
    """Transforms the pianoroll tensor into chroma features (not normalized).

    Args:
        pianoroll: Input pianoroll tensor,
            sized `[batch_size, bars, beat_resolution*4, pitch_span, num_tracks]`.
    """
    if pianoroll.get_shape().ndims != 5:
        raise ValueError("Input tensor must have 5 dimensions.")

    remainder = pianoroll.get_shape()[3] % 12
    if remainder > 0:
        pianoroll = tf.pad(
            pianoroll,
            ((0, 0), (0, 0), (0, 0), (0, 12 - remainder), (0, 0))
        )

    reshaped = tf.reshape(
        pianoroll,
        tf.concat([tf.shape(pianoroll)[:3], (12, pianoroll.get_shape()[3] // 12, pianoroll.get_shape()[4])], 0)
    )

    return tf.reduce_sum(reshaped, 4)


# Musical metrics
def tf_empty_bar_rate(tensor):
    """Computes the ratio of empty bars for each track of input pianoroll tensor.

    Args:
        tensor: Input pianoroll tensor,
            sized `[batch_size, bars, beat_resolution*4, pitch_span, num_tracks]`.

    Returns:
        empty_bar_rate: A tensor of empty bar ratios for each track.
    """
    if tensor.get_shape().ndims != 5:
        raise ValueError("Input tensor must have 5 dimensions.")

    return tf.reduce_mean(tf.cast(tf.reduce_all(tensor < 0.5, (2, 3)), tf.float32), (0, 1))


def tf_num_pitches_used(tensor):
    """Computes the average number of used pitches for each track of input pianoroll tensor.

    Args:
        tensor: Input pianoroll tensor,
            sized `[batch_size, bars, beat_resolution*4, pitch_span, num_tracks]`.

    Returns:
        num_pitches_used: A tensor of pitch utilization numbers for each track.
    """
    if tensor.get_shape().ndims != 5:
        raise ValueError("Input tensor must have 5 dimensions.")
    return tf.reduce_mean(
        tf.count_nonzero(tf.cast(tf.reduce_any(tensor > 0.5, 2), tf.float32), 2, dtype=tf.float32),
        (0, 1))


def tf_qualified_note_rate(tensor, threshold=2):
    """Computes the ratio of qualified notes for each track of input pianoroll tensor.

    Args:
        tensor: Input bar pianoroll tensor,
            sized `[batch_size, bars, beat_resolution*4, pitch_span, num_tracks]`.
        threshold: The minimum length of the qualified note in time steps.

    Returns:
        qualified_note_rate: A tensor of qualified note ratios for each track.
    """
    if tensor.get_shape().ndims != 5:
        raise ValueError("Input tensor must have 5 dimensions.")

    return tf.py_function(
        lambda array: qualified_note_rate(array.numpy(), threshold),
        [tensor], tf.float32
    )


def tf_polyphonic_rate(tensor, threshold=2):
    """Computes the ratio of polyphonic time steps for each track of input pianoroll tensor.

    Args:
        tensor: Input bar pianoroll tensor,
            sized `[batch_size, bars, beat_resolution*4, pitch_span, num_tracks]`.
        threshold: The minimum number of pitches in a polyphonic time step.

    Returns:
        polyphonic_rate: A tensor of polyphonic time step ratios for each track.
    """
    if tensor.get_shape().ndims != 5:
        raise ValueError("Input tensor must have 5 dimensions.")

    n_poly = tf.count_nonzero((tf.count_nonzero(tensor, 3) > threshold), 2)
    return tf.reduce_mean((n_poly / tensor.get_shape()[2]), [0, 1])


def tf_drum_in_pattern_rate(tensor):
    """Computes the ratio of drum notes in pattern in input drum pianoroll tensor.

    Args:
        tensor: Input drums chroma features,
            sized `[batch_size, bars, beat_resolution*4, 12]`.

    Returns:
        drum_in_pattern_rate: The ratio of drums in pattern.
    """
    if tensor.get_shape().ndims != 4:
        raise ValueError("Input tensor must have 4 dimensions.")

    return tf.py_function(
        lambda array: drum_in_pattern_rate(array.numpy()),
        [tensor], tf.float32
    )


def tf_harmonicity(tensor):
    """Computes the harmonicity (tonal distance [1]) between pairs of tracks.

    [1]: Christopher Harte, Mark Sandler, and Martin Gasser. Detecting
        harmonic change in musical audio. In Proc. ACM MM Workshop on Audio and
        Music Computing Multimedia, 2006.

    Args:
        tensor: Input chroma features for bar pianoroll tensor,
            sized `[batch_size, bars, beat_resolution*4, 12, num_tracks]`.

    Returns:
        tonal_distance: A tensor of track-to-track tonal distances,
            sized `[num_tracks x num_tracks]`.
    """
    if tensor.get_shape().ndims != 5:
        raise ValueError("Input tensor must have 5 dimensions.")
    if tensor.get_shape()[3] != 12:
        raise ValueError("Input tensor must be a chroma tensor.")

    return tf.py_function(
        lambda array: harmonicity(array.numpy()),
        [tensor], tf.float32
    )


# Metrics ops
def get_metric_ops(tensor):
    """Builds a dictionary of musical metric ops.

    Args:
        tensor: Input pianoroll tensor,
            sized `[batch_size, bars, beat_resolution*4, pitch_span, num_tracks]`.

    Returns:
        metric_ops: A dictionary of musical metric ops.
    """
    if tensor.get_shape().ndims != 5:
        raise ValueError("Input tensor must have 5 dimensions.")

    chroma = tf_to_chroma(tensor)

    metric_ops = {
        'EB': tf_empty_bar_rate(tensor),
        'UP': tf_num_pitches_used(tensor),
        'UPC': tf_num_pitches_used(chroma),
        'PR': tf_polyphonic_rate(tensor),
        'QN': tf_qualified_note_rate(tensor),
        'DP': tf_drum_in_pattern_rate(tensor[..., 0]),
        'TD': tf_harmonicity(chroma[..., 1:]),
    }

    return metric_ops


def get_metric_summary_ops(tensor, tracks):
    """Builds summaries for musical metric.

    Args:
        tensor: Input pianoroll tensor,
            sized `[batch_size, bars, beat_resolution*4, pitch_span, num_tracks]`.
        tracks: A list of track names for the input tensor.

    Returns:
        metric_ops: A tf.Summary object for musical metric.
    """
    # Getting metric ops
    with tf.variable_scope("musical_metrics/"):
        metric_ops = get_metric_ops(tensor)

    # Building summaries
    with tf.variable_scope("sample_scores/"):
        with tf.variable_scope("intra-track"):
            summaries = []
            for i in range(len(tracks)):
                track = tracks[i]
                with tf.variable_scope(f"{track}"):
                    summaries += [
                        tf.summary.scalar('EB', metric_ops['EB'][i]),
                        tf.summary.scalar('UP', metric_ops['UP'][i]),
                    ]
                    if track == 'Drums':
                        summaries.append(tf.summary.scalar('DP', metric_ops['DP']))
                    else:
                        summaries += [
                            tf.summary.scalar('UPC', metric_ops['UPC'][i]),
                            tf.summary.scalar('QN', metric_ops['QN'][i]),
                            tf.summary.scalar('PR', metric_ops['PR'][i]),
                        ]

        with tf.variable_scope("inter-track"):
            for i in range(1, len(tracks)):
                for j in range(i + 1, len(tracks)):
                    with tf.variable_scope(f"TD/{tracks[i]}-{tracks[j]}"):
                        summaries.append(tf.summary.scalar(f'{tracks[i]}-{tracks[j]}', metric_ops['TD'][i - 1][j - 1]))

        sample_summary = tf.summary.merge(summaries)

    return sample_summary
