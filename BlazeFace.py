import tensorflow as tf
import numpy as np


def build_network(img_size: tuple, channels=3):
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, *img_size, channels), name='inputs')
    phrase_train = tf.placeholder(dtype=tf.bool, name='phase_train')
    with tf.variable_scope('blaze_fafce'):
        with tf.variable_scope('first_conv'):
            # pad_inputs = tf.pad(inputs, [[0,0],[1,1],[1,1],[0,0]], mode='CONSTANT')
            conv1 = tf.layers.conv2d(inputs, 24, kernel_size=3, strides=2, padding='SAME')
            conv1 = tf.layers.batch_normalization(conv1, training=phrase_train, momentum=24.)
            conv1 = tf.nn.relu(conv1)
        with tf.variable_scope('blaze_block'):
            bb1 = blaze_block(conv1, filters=24, phase_train=phrase_train)
            bb1 = blaze_block(bb1, filters=24, phase_train=phrase_train)
            bb1 = blaze_block(bb1, filters=48, stride=2, phase_train=phrase_train)
            bb1 = blaze_block(bb1, filters=48, phase_train=phrase_train)
            bb1 = blaze_block(bb1, filters=48, phase_train=phrase_train)

        with tf.variable_scope('double_blaze'):
            db1 = double_blaze_block(bb1, filters=96, mid_channels=24, stride=2, phase_train=phrase_train)
            db1 = double_blaze_block(db1, filters=96, mid_channels=24, phase_train=phrase_train)
            db1 = double_blaze_block(db1, filters=96, mid_channels=24, phase_train=phrase_train)
            db1 = double_blaze_block(db1, filters=96, mid_channels=24, stride=2, phase_train=phrase_train)
            db1 = double_blaze_block(db1, filters=96, mid_channels=24, phase_train=phrase_train)
            db1 = double_blaze_block(db1, filters=96, mid_channels=24, phase_train=phrase_train)
            db1 = double_blaze_block(db1, filters=96, mid_channels=24, phase_train=phrase_train)
    return inputs, phrase_train, tf.identity(db1, 'output')


def blaze_block(x: tf.Tensor, filters, mid_channels=None, stride=1, phase_train=True):
    # input is n,w,h,c
    mid_channels = mid_channels or x.get_shape()[3]
    assert stride in [1, 2]
    use_pool = stride > 1
    # tensorflow way to implement pad size = 2
    pad_x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT')
    conv1 = tf.layers.separable_conv2d(pad_x, filters=mid_channels, kernel_size=(5, 5), strides=stride, padding='VALID')
    bn1 = tf.layers.batch_normalization(conv1, training=phase_train)
    conv2 = tf.layers.conv2d(bn1, filters=filters, kernel_size=1, strides=1, padding='SAME')
    bn2 = tf.layers.batch_normalization(conv2, training=phase_train)

    if use_pool:
        shortcut = tf.layers.max_pooling2d(x, pool_size=stride, strides=stride, padding='SAME')
        shortcut = tf.layers.conv2d(shortcut, filters=filters, kernel_size=1, strides=1, padding='SAME')
        shortcut = tf.layers.batch_normalization(shortcut, training=phase_train)
        shortcut = tf.nn.relu(shortcut)
        return tf.nn.relu(bn2 + shortcut)
    return tf.nn.relu(bn2 + x)


def double_blaze_block(x: tf.Tensor, filters, mid_channels=None, stride=1, phase_train=True):
    assert stride in [1, 2]
    mid_channels = mid_channels or x.get_shape()[3]
    usepool = stride > 1

    # padding = 2
    pad_x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT')
    conv1 = tf.layers.separable_conv2d(pad_x, filters=filters, kernel_size=5, strides=stride, padding='VALID')
    bn1 = tf.layers.batch_normalization(conv1, training=phase_train)
    conv1 = tf.layers.conv2d(bn1, filters=mid_channels, kernel_size=1, strides=1, padding='SAME')
    bn2 = tf.layers.batch_normalization(conv1, training=phase_train)
    relu1 = tf.nn.relu(bn2)

    # padding = 2
    pad_relu1 = tf.pad(relu1, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT')
    conv2 = tf.layers.separable_conv2d(pad_relu1, filters=mid_channels, kernel_size=5, strides=1, padding='VALID')
    bn2 = tf.layers.batch_normalization(conv2, training=phase_train)
    conv2 = tf.layers.conv2d(bn2, filters=filters, kernel_size=1, strides=1, padding='SAME')
    bn2 = tf.layers.batch_normalization(conv2, training=phase_train)

    # if use pool:
    if usepool:
        max_pool1 = tf.layers.max_pooling2d(x, pool_size=stride, strides=stride, padding='SAME')
        conv3 = tf.layers.conv2d(max_pool1, filters=filters, kernel_size=1, strides=1, padding='SAME')
        bn3 = tf.layers.batch_normalization(conv3, training=phase_train)
        return tf.nn.relu(bn2 + bn3)

    return tf.nn.relu(bn2 + x)


if __name__ == '__main__':
    inputs, phrase_train, output = build_network((128, 128))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(output, feed_dict={inputs: np.random.random((16, 128, 128, 3)), phrase_train: True})
