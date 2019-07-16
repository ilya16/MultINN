"""Auxiliary utility functions that do not fall under one identifiable group."""

import errno
import os

import tensorflow as tf


def safe_log(tensor):
    """Lower bounded log function."""
    return tf.log(1e-6 + tensor)


def time_to_str(time):
    """Converts time in seconds into pretty str."""
    return f'{int(time / 60.)}:{(time % 60.):.2f}'


def make_sure_dir_exists(path):
    """Creates intermediate directories if the path does not exist."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_current_scope():
    """Returns current TF scope."""
    current_name_scope = tf.get_default_graph().get_name_scope()
    current_name_scope += '/' if current_name_scope else ''
    return current_name_scope
