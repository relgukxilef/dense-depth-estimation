import tensorflow as tf

def depth_loss_function(y_true, y_pred):
    depth = y_true[:, :, :, 0]
    mask = y_true[:, :, :, 1]
    depth_pred = y_pred[:, :, :, 0]
    depth2_pred = y_pred[:, :, :, 1]

    variance = tf.squared_difference(depth, depth_pred)
    variance2 = tf.squared_difference(tf.square(depth), depth2_pred)

    return tf.reduce_mean((variance + variance2) * mask)

def depth_variance(y_true, y_pred):
    depth = y_true[:, :, :, 0]
    mask = y_true[:, :, :, 1]
    depth_pred = y_pred[:, :, :, 0]

    variance = tf.squared_difference(depth, depth_pred)

    return tf.reduce_sum(variance * mask) / tf.reduce_sum(mask)

def error_variance(y_true, y_pred):
    depth = y_true[:, :, :, 0]
    mask = y_true[:, :, :, 1]
    depth_pred = y_pred[:, :, :, 0]
    depth2_pred = y_pred[:, :, :, 1]

    variance = tf.squared_difference(depth, depth_pred)
    variance_pred = tf.maximum(0.0, depth2_pred - depth_pred * depth_pred)
    variance_variance = tf.squared_difference(
        tf.sqrt(variance), tf.sqrt(variance_pred)
    )

    return tf.reduce_sum(variance_pred * mask) / tf.reduce_sum(mask)