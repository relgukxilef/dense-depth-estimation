import tensorflow as tf

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
    depth = y_true[:, :, :, 0]
    mask = y_true[:, :, :, 1]
    depth_pred = y_pred[:, :, :, 0]
    variance_pred = y_pred[:, :, :, 1]
    #mask = 1

    variance = tf.squared_difference(depth, depth_pred)
    sum_mask = tf.reduce_sum(mask)

    # Point-wise depth
    l_depth = tf.reduce_sum(variance * mask) / sum_mask
    # TODO: predict relative error
    l_variance = tf.reduce_sum(
        tf.squared_difference(tf.stop_gradient(variance), variance_pred) * mask
    ) / sum_mask
    return l_depth + l_variance

    # Edges
    #dy_true, dx_true = tf.image.image_gradients(depth)
    #dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    #l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    #l_ssim = K.clip((1 - tf.image.ssim(depth, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta

    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))