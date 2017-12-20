import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

############ Generator ##############
def build_generator(inputs):
    with tf.variable_scope('generator') as scope:
        #inputs = tf.placeholder(tf.float32, [None, 123]) # 100 z~N(0,1) + 23 tags one-hot

        initializer = tf.random_normal_initializer(stddev=0.02)

        fc = tc.layers.fully_connected(inputs, 4*4*256, weights_initializer=initializer, activation_fn=None)
        fc = tf.layers.batch_normalization(fc)
        fc = tf.reshape(fc, [-1, 4, 4, 256])
        fc = tf.nn.relu(fc)

        conv1 = tc.layers.convolution2d_transpose(fc, 128, [5, 5], [2, 2], padding='same', weights_initializer=initializer, activation_fn=None)
        conv1 = tf.layers.batch_normalization(conv1)
        conv1 = tf.nn.relu(conv1)

        conv2 = tc.layers.convolution2d_transpose(conv1, 64, [5, 5], [2, 2], padding='same', weights_initializer=initializer, activation_fn=None)
        conv2 = tf.layers.batch_normalization(conv2)
        conv2 = tf.nn.relu(conv2)

        conv3 = tc.layers.convolution2d_transpose(conv2, 32, [5, 5], [2, 2], padding='same', weights_initializer=initializer, activation_fn=None)
        conv3 = tf.layers.batch_normalization(conv3)
        conv3 = tf.nn.relu(conv3)

        conv4 = tc.layers.convolution2d_transpose(conv3, 3, [5, 5], [2, 2], padding='same', weights_initializer=initializer, activation_fn=None)
        conv4 = tf.nn.tanh(conv4)

    # vars
    g_vars = [var for var in tf.global_variables() if "generator" in var.name]
    
    return conv4, g_vars

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def build_discriminator(img, seq, reuse=True):
    with tf.variable_scope('discriminator') as scope:
        #img = tf.placeholder(tf.float32, [None, 64, 64, 3])
        #seq = tf.placeholder(tf.float32, [None, 23])
        if reuse == True:
            scope.reuse_variables()

        initializer = tf.random_normal_initializer(stddev=0.02)

        conv1 = tc.layers.convolution2d(img, 32, [5, 5], [2, 2], padding='same', weights_initializer=initializer, activation_fn=None)
        conv1 = tf.layers.batch_normalization(conv1)
        conv1 = leaky_relu(conv1)

        conv2 = tc.layers.convolution2d(conv1, 64, [5, 5], [2, 2], padding='same', weights_initializer=initializer, activation_fn=None)
        conv2 = tf.layers.batch_normalization(conv2)
        conv2 = leaky_relu(conv2)

        conv3 = tc.layers.convolution2d(conv2, 128, [5, 5], [2, 2], padding='same', weights_initializer=initializer, activation_fn=None)
        conv3 = tf.layers.batch_normalization(conv3)
        conv3 = leaky_relu(conv3)

        tag_vector = tf.expand_dims(tf.expand_dims(seq, 1), 2)
        tag_vector = tf.tile(tag_vector, [1, 8, 8, 1])
        condition_info = tf.concat([conv3, tag_vector], axis=-1)

        conv4 = tc.layers.convolution2d(condition_info, 128, [1, 1], [1, 1], padding='same', weights_initializer=initializer, activation_fn=None)
        conv4 = tf.layers.batch_normalization(conv4)
        conv4 = leaky_relu(conv4)

        conv5 = tc.layers.convolution2d(conv4, 1, [8, 8], [1, 1], padding='valid', weights_initializer=initializer, activation_fn=None)
        output = tf.squeeze(conv5, [1, 2, 3])

    # vars
    d_vars = [var for var in tf.global_variables() if "discriminator" in var.name]
    
    return output, d_vars









