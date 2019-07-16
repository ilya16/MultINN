"""Implementation of the musical metrics for evaluating the music samples.

The metrics are implemented using NumPy.

The shape of music is `[batch_size, bars, beat_resolution*4, pitch_span, num_tracks]`.

Note:
    The metrics are adapted from: https://salu133445.github.io/pypianoroll/
"""

import numpy as np


# Utility functions
def _to_chroma(pianoroll):
    """Transforms pianorolls into chroma scale arrays

    Args:
        pianoroll: Input bar pianorolls,
            sized `[batch_size, bars, beat_resolution*4, pitch_span, num_tracks]`.

    Returns:
        chroma_pianoroll: Chroma features pianorolls,
            sized [batch_size, bars, beat_resolution*4, 12, num_tracks].
    """
    remainder = pianoroll.shape[-2] % 12

    if remainder > 0:
        zero_pad = [(0, 0) for _ in range(len(pianoroll.shape) - 2)]
        pianoroll = np.pad(
            pianoroll,
            zero_pad + [(0, 12 - remainder), (0, 0)],
            'constant'
        )

    reshaped = np.reshape(
        pianoroll,
        pianoroll.shape[:-2] + (12, pianoroll.shape[-2] // 12, pianoroll.shape[-1])
    )

    return reshaped.sum(axis=-2)


# Musical metrics
def empty_bar_rate(pianoroll):
    """Computes the ratio of empty bars for each track of input pianorolls.

    Args:
        pianoroll: Input bar pianorolls,
            sized `[batch_size, bars, beat_resolution*4, pitch_span, num_tracks]`.

    Returns:
        empty_bar_rate: A list of empty bar ratios for each track.
    """
    if len(pianoroll.shape) != 5:
        raise ValueError("Input tensor must have 5 dimensions.")

    return 1 - pianoroll.any(axis=(2, 3)).mean(axis=(0, 1))


def num_pitches_used(pianoroll):
    """Computes the average number of used pitches for each track of input pianorolls.

    Args:
        pianoroll: Input bar pianorolls,
            sized `[batch_size, bars, beat_resolution*4, pitch_span, num_tracks]`.

    Returns:
        num_pitches_used: A list of pitch utilization numbers for each track.
    """
    if len(pianoroll.shape) != 5:
        raise ValueError("Input tensor must have 5 dimensions.")

    return np.count_nonzero(np.any(pianoroll, 2), axis=2).mean(axis=(0, 1))


def qualified_note_rate(pianoroll, threshold=2):
    """Computes the ratio of qualified notes for each track of input pianorolls.

    Args:
        pianoroll: Input bar pianorolls,
            sized `[batch_size, bars, beat_resolution*4, pitch_span, num_tracks]`.
        threshold: The minimum length of the qualified note in time steps.

    Returns:
        qualified_note_rate: A list of qualified note ratios for each track.
    """
    if len(pianoroll.shape) != 5:
        raise ValueError("Input tensor must have 5 dimensions.")

    num_tracks = pianoroll.shape[-1]

    reshaped = pianoroll.reshape((-1, pianoroll.shape[1] * pianoroll.shape[2],) + pianoroll.shape[3:])
    padded = np.pad(reshaped, ((0, 0), (1, 1), (0, 0), (0, 0)), 'constant')
    diff = np.diff(padded.astype(np.int32), axis=1)
    del padded, reshaped

    transposed = diff.transpose((3, 0, 2, 1)).reshape(num_tracks, -1)
    onsets = (transposed > 0).nonzero()
    offsets = (transposed < 0).nonzero()

    n_qualified_notes = np.array(
        [np.count_nonzero(offsets[1][(offsets[0] == i)] - onsets[1][(onsets[0] == i)] > threshold)
         for i in range(num_tracks)], np.float32
    )

    n_onsets = np.array(
        [np.count_nonzero(onsets[1][(onsets[0] == i)]) for i in range(num_tracks)],
        np.float32
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        return n_qualified_notes / n_onsets


def polyphonic_rate(pianoroll, threshold=2):
    """Computes the ratio of polyphonic time steps for each track of input pianorolls.

    Args:
        pianoroll: Input bar pianorolls,
            sized `[batch_size, bars, beat_resolution*4, pitch_span, num_tracks]`.
        threshold: The minimum number of pitches in a polyphonic time step.

    Returns:
        polyphonic_rate: A list of polyphonic time step ratios for each track.
    """
    if len(pianoroll.shape) != 5:
        raise ValueError("Input tensor must have 5 dimensions.")

    n_poly = np.count_nonzero(np.count_nonzero(pianoroll, 3) > threshold, 2)

    return (n_poly / pianoroll.shape[2]).mean(axis=(0, 1))


def drum_in_pattern_rate(chroma):
    """Computes the ratio of drum notes in pattern in input drum pianorolls.

    Args:
        chroma: Input drums chroma features,
            sized `[batch_size, bars, beat_resolution*4, 12]`.

    Returns:
        drum_in_pattern_rate: The ratio of drums in pattern.
    """
    if len(chroma.shape) != 4:
        raise ValueError("Input tensor must have 4 dimensions.")

    def _drum_pattern_mask(n_timesteps, tolerance=0.1):
        """Return a drum pattern mask with the given tolerance."""
        if n_timesteps == 96:
            return np.tile([1., tolerance, 0., 0., 0., tolerance], 16)
        elif n_timesteps == 48:
            return np.tile([1., tolerance, tolerance], 16)
        elif n_timesteps == 24:
            return np.tile([1., tolerance, tolerance], 8)
        elif n_timesteps == 72:
            return np.tile([1., tolerance, 0., 0., 0., tolerance], 12)
        elif n_timesteps == 36:
            return np.tile([1., tolerance, tolerance], 12)
        elif n_timesteps == 64:
            return np.tile([1., tolerance, 0., tolerance], 16)
        elif n_timesteps == 32:
            return np.tile([1., tolerance], 16)
        elif n_timesteps == 16:
            return np.tile([1., tolerance], 8)
        else:
            raise ValueError("Unsupported number of timesteps for the drum in pattern metric.")

    drum_pattern_mask = _drum_pattern_mask(chroma.shape[2])
    drum_pattern_mask = drum_pattern_mask.reshape((1, 1, chroma.shape[2]))

    num_in_pattern = np.sum(drum_pattern_mask * np.sum(chroma, 3))
    num_notes = np.count_nonzero(chroma)

    return num_in_pattern / num_notes if num_notes > 0 else 0.


def harmonicity(chroma):
    """Computes the harmonicity (tonal distance [1]) between pairs of tracks.

    [1]: Christopher Harte, Mark Sandler, and Martin Gasser. Detecting
        harmonic change in musical audio. In Proc. ACM MM Workshop on Audio and
        Music Computing Multimedia, 2006.

    Args:
        chroma: Input chroma features for bar pianorolls,
            sized `[batch_size, bars, beat_resolution*4, 12, num_tracks]`.

    Returns:
        tonal_distance: A matrix of track-to-track tonal distances,
            sized `[num_tracks x num_tracks]`.
    """
    if len(chroma.shape) != 5:
        raise ValueError("Input tensor must have 5 dimensions.")
    if chroma.shape[3] != 12:
        raise ValueError("Input tensor must be a chroma tensor.")

    def _tonal_matrix(r1=1.0, r2=1.0, r3=0.5):
        """Returns a tonal matrix for computing the tonal distance [1].

        Default argument values are set as suggested by the paper.
        """
        tonal_matrix = np.empty((6, 12))
        tonal_matrix[0] = r1 * np.sin(np.arange(12) * (7. / 6.) * np.pi)
        tonal_matrix[1] = r1 * np.cos(np.arange(12) * (7. / 6.) * np.pi)
        tonal_matrix[2] = r2 * np.sin(np.arange(12) * (3. / 2.) * np.pi)
        tonal_matrix[3] = r2 * np.cos(np.arange(12) * (3. / 2.) * np.pi)
        tonal_matrix[4] = r3 * np.sin(np.arange(12) * (2. / 3.) * np.pi)
        tonal_matrix[5] = r3 * np.cos(np.arange(12) * (2. / 3.) * np.pi)

        return tonal_matrix

    def _to_tonal_space(tensor):
        """Returns the tensor in tonal space where chroma features are normalized per beat."""
        tonal_matrix = _tonal_matrix()

        beat_chroma = tensor.reshape(-1, tensor.shape[2] // 4, 12, tensor.shape[4]).sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            beat_chroma = beat_chroma / beat_chroma.sum(axis=1, keepdims=True)
        reshaped = np.reshape(np.transpose(beat_chroma, (1, 0, 2)), (12, -1))

        return np.reshape(np.matmul(tonal_matrix, reshaped), (6, -1, tensor.shape[4]))

    mapped = _to_tonal_space(chroma)
    expanded1 = np.expand_dims(mapped, -1)
    expanded2 = np.expand_dims(mapped, -2)
    tonal_dist = np.linalg.norm(expanded1 - expanded2, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nanmean(tonal_dist, axis=0)


# Sample evaluation
def compute_sample_metrics(pianoroll):
    """Evaluates sample bar pianorolls and outputs results to console.

    Args:
        pianoroll: Input bar pianorolls,
            sized `[batch_size, bars, beat_resolution*4, pitch_span, num_tracks]`.
    """
    if len(pianoroll.shape) != 5:
        raise ValueError("Input tensor must have 5 dimensions.")

    chroma = _to_chroma(pianoroll[..., 1:])

    print()
    print(' ' * 5 + ' Drums    Piano    Guitar   Bass    Strings')

    print(f'{"EB: ":5s}', end='')
    ebr = empty_bar_rate(pianoroll)
    print('  '.join(map(lambda x: f'{x:.5f}', ebr)))

    print(f'{"UP: ":5s}', end='')
    up = num_pitches_used(pianoroll)
    print('  '.join(map(lambda x: f'{x:.5f}', up)))

    print(f'{"UPC: ":5s}   -     ', end='')
    upc = num_pitches_used(chroma)
    print('  '.join(map(lambda x: f'{x:.5f}', upc)))

    print(f'{"QN: ":5s}   -     ', end='')
    for i in range(1, 5):
        qn = qualified_note_rate(pianoroll[..., i:i + 1])[0]
        print(f'{qn:.5f}  ', end='')

    print(f'\n{"PR: ":5s}   -     ', end='')
    pr = polyphonic_rate(pianoroll[..., 1:])
    print('  '.join(map(lambda x: f'{x:.5f}', pr)))

    dp = drum_in_pattern_rate(pianoroll[..., 0])
    print(f'\n{"DP: ":5s}{dp:.5f}')
    del pianoroll

    print(f'\n{"TD: ":5s}')
    print(harmonicity(chroma))
