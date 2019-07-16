"""Implementation of the statistical metrics for evaluating the models."""

import tensorflow as tf


def base_metrics(loss, targets, predictions, log_probs):
    """Builds base metrics for models based on targets and predictions.

    Args:
        loss: A list of model's losses, sized `[batch_size]`.
        targets: Flattened target predictions,
            sized `[batch_size, num_dims]`.
        predictions: Flattened predictions,
            sized `[batch_size, num_dims]`.
        log_probs: A list of log probabilities for predictions,
            sized `[batch_size]`.

    Returns:
        metrics: A dictionary of metric names and metric ops.
        metrics_upd: A list of metric update ops.
    """
    perplexity = tf.exp(log_probs)

    metrics, metrics_upd = tf.contrib.metrics.aggregate_metric_map(
        {
            'loss': tf.metrics.mean(loss, name='loss_op'),
            'log_likelihood': tf.metrics.mean(log_probs, name='log_likelihood_op'),
            'perplexity': tf.metrics.mean(perplexity, name='perplexity_op'),
            'accuracy': tf.metrics.accuracy(targets, predictions, name='accuracy_op'),
            'precision': tf.metrics.precision(targets, predictions, name='precision_op'),
            'recall': tf.metrics.recall(targets, predictions, name='recall_op'),
        })

    metrics['batch/loss'] = tf.reduce_mean(loss)

    with tf.variable_scope('f1_score_op'):
        precision = metrics['precision']
        recall = metrics['precision']
        f1_score = tf.where(
            tf.greater(precision + recall, 0),
            2 * ((precision * recall) / (precision + recall)), 0)
        metrics['f1_score'] = f1_score

    for update_op in metrics_upd.values():
        tf.add_to_collection('eval_op', update_op)

    return metrics, list(metrics_upd.values())


def build_summaries(metrics):
    """Builds summaries from metrics.

    Args:
        metrics: A dictionary of metric names and metric ops.

    Returns:
        summaries: A tf.Summary object for model's metrics.
    """
    summaries = [
        tf.summary.scalar(var_name, var_value)
        for var_name, var_value in metrics.items()
        if var_name != 'batch/loss'
    ]

    return tf.summary.merge(summaries)
