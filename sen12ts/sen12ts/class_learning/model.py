import tensorflow as tf


def last_layer(filters, size, strides, padding):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    regularizer = tf.keras.regularizers.L2(l2=0.01)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=size,
            strides=strides,
            padding=padding,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=False,
            name='last_layer_with_softmax'
        )
    )

    result.add(tf.keras.layers.Softmax())

    return result


def conv2d_layer(filters, size, strides, padding, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    regularizer = tf.keras.regularizers.L2(l2=0.01)
    
    
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=size,
            strides=strides,
            padding=padding,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    # result.add(tf.keras.layers.ELU())

    return result

def conv2dtranspose_layer(filters, size, strides, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    regularizer = tf.keras.regularizers.L2(l2=0.01)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=size,
                strides=strides,
                padding='same',
                kernel_initializer=initializer,
                kernel_regularizer=regularizer,
                use_bias=False,
        )
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def ts_conv_layer(imagery_list_by_ts, filters, size, strides, padding='same'):

    output_list = []
    conv_layer = conv2d_layer(filters=filters, size=size, strides=strides, padding=padding)

    for image in imagery_list_by_ts:
        output_list.append(conv_layer(image))

    return output_list

def ts_max_pool(imagery_list_by_ts, pool_size):

    output_list = []
    max_pool_layer = tf.keras.layers.MaxPool2D(pool_size=pool_size, padding='valid')

    for image in imagery_list_by_ts:
        output_list.append(max_pool_layer(image))

    return output_list


def average_across_ts(imagery_list_by_ts):

    imagery_list_by_ts = tf.keras.layers.Average()(imagery_list_by_ts)
    
#     return imagery_list_by_ts[0]
    return imagery_list_by_ts

def padding_2d_across_ts(imagery_list_by_ts, padding):
    output_list = []
    padding_2d_layer = tf.keras.layers.ZeroPadding2D(padding=padding)

    for image in imagery_list_by_ts:
        output_list.append(padding_2d_layer(image))

    return output_list


def conv2dlstm_layer_avg_and_reduce(concat_layer, filters, size):


    conv2dlstm = tf.keras.layers.ConvLSTM2D(filters=filters, kernel_size=size,
                                            padding='same',
                                            return_sequences=True)

    out = conv2dlstm(concat_layer)
    
    # Average across the temporal dimension
    out = tf.math.reduce_mean(out, axis=1)



    return out

def convlstm_network(args, imagery_type, apply_softmax=True):

    
    if imagery_type == 's1':
        depth =  args.S1_INPUT_CHANNELS
    elif imagery_type == 's2':
        depth = args.S2_INPUT_CHANNELS
    

    inp = tf.keras.layers.Input(
            shape=[args.NUM_TIMESTEPS, args.INPUT_HEIGHT, args.INPUT_WIDTH, depth]
    )
    inp_list = []

    for i in range(args.NUM_TIMESTEPS):
        inp_list.append(inp[:, i, ...])


    # Top level of UNET given 64 x 64 images
    l2_imagery_list = ts_conv_layer(inp_list, filters=32, size=3, strides=1) # [list of (bn, 64, 64, 32)]
    l2_imagery_list = ts_conv_layer(l2_imagery_list, filters=32, size=3, strides=1) # [list of (bn, 64, 64, 32)]
#     l2_imagery_list_skip = ts_max_pool(l2_imagery_list, pool_size=2) # [list of (bn, 32, 32, 32)]
    l2_imagery_list_skip = average_across_ts(l2_imagery_list)
    
    # Drop an imagery_list to level 3
    l3_imagery_list = ts_max_pool(l2_imagery_list, pool_size=2) # [list of (bn, 32, 32, 32)]
    l3_imagery_list = ts_conv_layer(l3_imagery_list, filters=64, size=3, strides=1) # [list of (bn, 32, 32, 64)]
    l3_imagery_list = ts_conv_layer(l3_imagery_list, filters=64, size=3, strides=1) # [list of (bn, 32, 32, 64)]
    l3_imagery_list_skip = average_across_ts(l3_imagery_list)
    
    # Drop an imagery_list to level 4
    l4_imagery_list = ts_max_pool(l3_imagery_list, pool_size=2)  # [list of (bn, 16, 16, 64)]
    l4_imagery_list = ts_conv_layer(l4_imagery_list, filters=128, size=3, strides=1)  # [list of (bn, 16, 16, 128)]
    l4_imagery_list = ts_conv_layer(l4_imagery_list, filters=128, size=3, strides=1)  # [list of (bn, 16, 16, 128)]
    l4_imagery_list_skip = average_across_ts(l4_imagery_list)

    # Drop an imagery_list to level 5
    l5_imagery_list = ts_max_pool(l4_imagery_list, pool_size=2)  # [list of (bn, 8, 8, 128)]
    l5_imagery_list = ts_conv_layer(l5_imagery_list, filters=256, size=3, strides=1)  # [list of (bn, 8, 8, 256)]

  
    # Concatenate
    concat_layer = tf.keras.layers.Concatenate(axis=0)([tf.expand_dims(i, axis=0) for i in l5_imagery_list])
#     concat_layer = tf.expand_dims(l5_imagery_list[0], axis=0)
    # Reorder dims so that time dimension is second and overall dims are: (samples, time, rows, cols, channels)
    concat_layer = tf.transpose(concat_layer, perm=[1,0,2,3,4])

    # Input concatenated layer to LSTM + channel-reducing layers; convolve
    lstm_output = conv2dlstm_layer_avg_and_reduce(concat_layer, filters=256, size=3)
    l5_conv_output = conv2d_layer(filters=256, size=3, strides=1, padding='same')(lstm_output)
    
    # Apply transpose convolution to raise to level 4; concatenate; convolved
    l4_conv_output = conv2dtranspose_layer(filters=128, size=3, strides=2)(l5_conv_output)
    l4_conv_output = tf.keras.layers.Concatenate(axis=-1)([l4_imagery_list_skip, l4_conv_output])
    l4_conv_output = conv2d_layer(filters=128, size=3, strides=1, padding='same')(l4_conv_output)
    l4_conv_output = conv2d_layer(filters=128, size=3, strides=1, padding='same')(l4_conv_output)
    
     # Apply transpose convolution to raise to level 3; concatenate; convolve
    l3_conv_output = conv2dtranspose_layer(filters=64, size=3, strides=2)(l4_conv_output)
    l3_conv_output = tf.keras.layers.Concatenate(axis=-1)([l3_imagery_list_skip, l3_conv_output])
    l3_conv_output = conv2d_layer(filters=64, size=3, strides=1, padding='same')(l3_conv_output)
#     l3_conv_output = conv2dtranspose_layer(filters=32, size=3, strides=1)(l3_conv_output)
    
     # Apply transpose convolution to raise to level 2; concatenate; convolve
    l2_conv_output = conv2dtranspose_layer(filters=32, size=3, strides=2)(l3_conv_output)
    l2_conv_output = tf.keras.layers.Concatenate(axis=-1)([l2_imagery_list_skip, l2_conv_output])
    l2_conv_output = conv2d_layer(filters=32, size=3, strides=1, padding='same')(l2_conv_output)
    
    
    
    model_output = conv2d_layer(filters=args.NUM_CLASSES, size=3, strides=1, padding='same')(l2_conv_output)
    
    # Apply activation function
    if apply_softmax:
        model_output = tf.nn.softmax(model_output, axis=-1)
    
    return tf.keras.Model(inputs=inp, outputs=model_output)


def stacked_s1s2_models(args):
    
    s1_network = convlstm_network(args, 's1', apply_softmax=False)
    s2_network = convlstm_network(args, 's2', apply_softmax=False)


    merged_output = tf.keras.layers.Add()([s1_network.output, s2_network.output])
    
    merged_output = tf.keras.layers.Dense(args.NUM_CLASSES)(merged_output)
    
    merged_output = tf.nn.softmax(merged_output, axis=-1)
    
    combined_model = tf.keras.Model([s1_network.input, s2_network.input], merged_output)
    
    return combined_model

