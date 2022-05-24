import tensorflow as tf
# import numpy as np

def crossentropy_loss(model_output, target, class_weights):
    '''
    model_output and target are both tensors of shape: (batch, 256, 256, num_crops)
    '''

    # Extract pixels that correspond to one of top 10 classes
    valid_pixels = tf.math.equal(tf.reduce_max(target, axis=-1), 1)

    output_for_loss = model_output[valid_pixels]
    target_for_loss = target[valid_pixels]

    loss_fx = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    
    weights_full = tf.ones([tf.shape(output_for_loss)[0], 1], dtype=tf.float32) * class_weights #[..., 0:-1]
    weights = tf.cast(target_for_loss, tf.float32) * weights_full
    weights = tf.reduce_max(weights, axis=-1)

    
    loss = loss_fx(target_for_loss, output_for_loss, sample_weight=weights)
    valid_loss = loss[~tf.math.is_nan(loss)]
    
    mean_loss = tf.math.reduce_mean(valid_loss)

    return mean_loss
