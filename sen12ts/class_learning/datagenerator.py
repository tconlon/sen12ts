import tensorflow as tf
import glob
import numpy as np
import os
import argparse, yaml

def get_args():
    parser = argparse.ArgumentParser(
        description="Image to image translation with Pix2Pix"
    )

    parser.add_argument(
        "--training_params_filename",
        type=str,
        default="params.yaml",
        help="Filename defining model configuration",
    )

    args = parser.parse_args()
    config = yaml.safe_load(open(args.training_params_filename))
    for k, v in config.items():
        args.__dict__[k] = v

    return args



def parse_example(example_proto):
    '''
    Helper function converts .tfrecords into input and output images
    '''

    sar2vi = False

    if sar2vi:
        dates = tf.io.FixedLenFeature([], tf.string)
    else:
        dates = tf.io.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True)

    features = {
        "image/image_data": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/channels": tf.io.FixedLenFeature([], tf.int64),
        "target/target_data": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),
        "target/height": tf.io.FixedLenFeature([], tf.int64),
        "target/width": tf.io.FixedLenFeature([], tf.int64),
        "target/channels": tf.io.FixedLenFeature([], tf.int64),
        "dates": dates,
        
    }

    image_features = tf.io.parse_single_example(example_proto, features)

    img_height = tf.cast(image_features["image/height"], tf.int32)
    img_width = tf.cast(image_features["image/width"], tf.int32)
    img_channels = tf.cast(image_features["image/channels"], tf.int32)

    target_height = tf.cast(image_features["target/height"], tf.int32)
    target_width = tf.cast(image_features["target/width"], tf.int32)
    target_channels = tf.cast(image_features["target/channels"], tf.int32)
    

    image_raw = tf.reshape(
        tf.squeeze(image_features["image/image_data"]),
        tf.stack([img_height, img_width, img_channels]),
    )

    target_raw = tf.reshape(
        tf.squeeze(image_features["target/target_data"]),
        tf.stack([target_height, target_width, target_channels]),
    )

    dates = image_features['dates']

    return image_raw, target_raw, dates

def type_transform(s1_image, s2_image, dates):
    '''
    Ensuring the input images are of type float 32
    and the target images are of type uint 8
    '''
    
    s1_image  = tf.cast(s1_image, tf.float32)
    s2_image = tf.cast(s2_image, tf.float32)

    return s1_image, s2_image, dates

def list_imagery_by_ts(s1_image, s2_image, dates):
    
    n_s2_bands = 13
    n_s1_bands = 4
    n_timesteps = 16
    
    s1_list = []
    s2_list = []
    
    for n in range(n_timesteps):
        s1_list.append(s1_image[..., n*n_s1_bands:(n+1)*n_s1_bands])
        s2_list.append(s2_image[..., n*n_s2_bands:(n+1)*n_s2_bands])

    usda_layer = s2_image[..., -1][..., None]

    return s1_list, s2_list, usda_layer, dates


def one_hot_encoding_target(args, s1_image, s2_image, dates):

    s2_image = tf.one_hot(s2_image, depth=args.NUM_CROPS, axis=-1)
    s2_image = tf.squeeze(s2_image, axis=-2)

    return s1_image, s2_image, dates



def encode_target_to_select_labels(label_df, s1_list, s2_list, usda_layer, dates):

    labels = label_df['label'].astype(np.uint16)
    usda_layer_list = []

    for ix, label in enumerate(labels):
        layer = tf.cast(tf.equal(usda_layer, label), tf.uint16)
        usda_layer_list.append(layer)

    usda_layer = tf.concat(usda_layer_list, axis=-1)

    # other_label_layer = tf.expand_dims(1 - tf.math.reduce_max(usda_layer, axis=-1), -1)
    # usda_layer = tf.concat((usda_layer, other_label_layer), axis=-1)

    return s1_list, s2_list, usda_layer, dates

