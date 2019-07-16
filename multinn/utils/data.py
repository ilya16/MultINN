"""Functions for working with music datasets."""
import os

import numpy as np
from pypianoroll import Multitrack, Track


def pad_to_midi(songs, data_config):
    """Pads and reshapes the songs into pianoroll format with 128 pitches.

    Args:
        songs: The songs in model's input format,
            sized `[batch_size, time_steps, step_span, num_tracks]`
        data_config: A data configuration dictionary.

    Returns:
        pianoroll: The padded pianoroll songs,
            sized `[batch_size, steps, 128, num_tracks]`
    """
    songs = np.reshape(
        songs,
        (songs.shape[0], -1, data_config['pitch_range']['highest'] - data_config['pitch_range']['lowest'],
         songs.shape[-1])
    )

    songs = np.pad(
        songs,
        ((0, 0), (0, 0), (data_config['pitch_range']['lowest'], 128 - data_config['pitch_range']['highest']), (0, 0)),
        'constant',
        constant_values=0
    )

    return songs


def load_data(data_config, step_size=1):
    """Loads the dataset from the local memory.

    Args:
        data_config: A data configuration dictionary.
        step_size: A size of one time step in pixels.
            Used to reshape the data for the models.

    Returns:
        (data_train, lengths_train): The training data,
            sized `[num_train, time_steps, num_dims, num_tracks]`,
            and training data lengths, sized `[num_train]`.
        (data_valid, lengths_valid): The validation data,
            sized `[num_valid, time_steps, num_dims, num_tracks]`,
            and validation data lengths, sized `[num_valid]`.
        (data_test, lengths_test): The test data,
            sized `[num_test, time_steps, num_dims, num_tracks]`,
            and test data lengths, sized `[num_test]`.
    """
    path = data_config['filename']

    num_train = data_config['split']['num_train']
    num_valid = data_config['split']['num_valid']
    num_test = data_config['split']['num_test']

    if data_config['source'] == 'npy':
        songs = np.load(f'{path}.npy')[:num_train + num_valid + num_test]

        if len(songs.shape) != 4:
            raise ValueError("Dataset must have 4 dimensions.")

        if songs.shape[-1] != len(data_config['instruments']):
            raise ValueError(f"Dataset must have {len(data_config['instruments'])} tracks.")

        # Padding the data if needed
        pad_size = step_size - (songs.shape[1] % step_size)
        pad_size = 0 if pad_size == step_size else pad_size
        if pad_size > 0:
            songs = np.pad(songs, ((0, 0), (0, pad_size), (0, 0), (0, 0)), 'constant', constant_values=0)

        songs = songs.reshape(
            [songs.shape[0], songs.shape[1] // step_size,
             songs.shape[2] * step_size, songs.shape[3]]
        )

        # Computing data sequences lengths
        if data_config['sequence_lengths']:
            lengths = np.load(data_config['sequence_lengths'])[:num_train + num_valid + num_test]
        else:
            lengths = np.full(songs.shape[0], songs.shape[1])
    else:
        raise ValueError('Not supported data format :(')

    # Dataset split
    data_train, lengths_train = songs[:num_train], lengths[:num_train]
    data_valid, lengths_valid = songs[num_train:num_train + num_valid], lengths[num_train:num_train + num_valid]
    data_test, lengths_test = songs[-num_test:], lengths[-num_test:]

    return (data_train, lengths_train), (data_valid, lengths_valid), (data_test, lengths_test)


def prepare_sampling_inputs(X_train, X_valid, sampling_config, beat_size):
    """Prepares inputs for the sampling based on the configurations.

    Args:
        X_train: The training data,
            sized `[num_train, time_steps, num_dims, num_tracks]`.
        X_valid: The validation data,
            sized `[num_valid, time_steps, num_dims, num_tracks]`.
        sampling_config: A sampling configuration config.
        beat_size: A beat size in time steps.

    Returns:
        intro_songs: The inputs to the sampling,
            sized `[num_sample_train+num_sample_valid, intro_steps, num_dims, num_tracks]`.
        save_ids: Ids of the sampled songs to save to disk.
        song_labels: Labels of song for saving.
    """
    intro_beats = sampling_config['intro_beats']
    intro_steps = int(intro_beats * beat_size)

    # Intro songs
    intro_ids = sampling_config['intro_ids']
    intro_train = X_train[intro_ids['train']['start']:intro_ids['train']['end'], :intro_steps, :]
    intro_valid = X_valid[intro_ids['valid']['start']:intro_ids['valid']['end'], :intro_steps, :]
    intro_songs = np.concatenate([intro_train, intro_valid], axis=0)
    del intro_train, intro_valid

    # Songs to save
    save_ids = sampling_config['save_ids']
    save_train = np.array(save_ids['train'])
    save_valid = np.array(save_ids['valid'])
    song_labels = [f't{i}' for i in save_train] + [f'v{i}' for i in save_valid]
    save_valid += intro_ids['train']['end'] - intro_ids['train']['start']

    save_ids = np.concatenate([save_train, save_valid], axis=0)

    next_ids = save_ids
    for i in range(1, sampling_config['num_save']):
        next_ids = next_ids + len(intro_songs)
        save_ids = np.concatenate([save_ids, next_ids], axis=0)

    return intro_songs, save_ids, song_labels


def write_song(song, path, data_config):
    """Writes multi-track pianoroll song into a MIDI file.

    Saves the song using the `pypianoroll` package.

    Args:
        song: A song pianoroll array,
            sized `[time_steps, 128, num_tracks]`.
        path: A path where the song should be saved.
        data_config: A data configuration dictionary.
    """
    song = song * 100.
    instruments = data_config['instruments']

    tracks = []
    for i in range(len(instruments)):
        name = instruments[i]
        track = np.take(song, i, axis=-1)

        # Manually adjusting the sound
        if name == 'Piano':
            track = track * 0.8
        elif name == 'Strings':
            track = track * 0.9
        elif name == 'Bass':
            track = track * 1.2

        track = Track(
            pianoroll=track,
            program=data_config['programs'][i],
            is_drum=data_config['is_drums'][i])
        tracks.append(track)

    mtrack = Multitrack(
        tracks=tracks,
        tempo=data_config['tempo'],
        beat_resolution=data_config['beat_resolution'])

    mtrack.write(path)


def save_music(music, num_intro, data_config, base_path, save_dir='outputs/', song_labels=None):
    """Saves a batch of songs into MIDI Files.

    Args:
        music: An array of songs in pianoroll format,
            sized `[num_songs x num_intro, time_steps, 128, num_tracks]`.
        num_intro: A number of unique intro songs.
        data_config: A data configuration dictionary.
        base_path: A base name for each song.
        save_dir: A directory for saving the samples.
        song_labels: A list of labels for each intro song.
    """

    for i in range(num_intro):
        for j in range(music.shape[0] // num_intro):
            if song_labels is None:
                song_path = os.path.join(save_dir, base_path + f'_song{i}_{j}.mid')  # The new song will be saved here
            else:
                song_path = os.path.join(save_dir, base_path + f'_{song_labels[i]}_{j}.mid')

            write_song(music[i + j * num_intro], song_path, data_config)
