import tensorflow as tf
import numpy as np


def build_backbone(img_size: tuple, channels=3):
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
            feature32by32 = double_blaze_block(db1, filters=96, mid_channels=24, phase_train=phrase_train)
            db2 = double_blaze_block(feature32by32, filters=96, mid_channels=24, stride=2, phase_train=phrase_train)
            db2 = double_blaze_block(db2, filters=96, mid_channels=24, phase_train=phrase_train)
            db2 = double_blaze_block(db2, filters=96, mid_channels=24, phase_train=phrase_train)
            feature16by16 = double_blaze_block(db2, filters=96, mid_channels=24, phase_train=phrase_train)
            db3 = double_blaze_block(feature16by16, filters=96, mid_channels=24, stride=2, phase_train=phrase_train)
            db3 = double_blaze_block(db3, filters=96, mid_channels=24, phase_train=phrase_train)
            db3 = double_blaze_block(db3, filters=96, mid_channels=24, phase_train=phrase_train)
            feature8by8 = double_blaze_block(db3, filters=96, mid_channels=24, phase_train=phrase_train)
            db4 = double_blaze_block(feature8by8, filters=96, mid_channels=24, stride=2, phase_train=phrase_train)
            db4 = double_blaze_block(db4, filters=96, mid_channels=24, phase_train=phrase_train)
            db4 = double_blaze_block(db4, filters=96, mid_channels=24, phase_train=phrase_train)
            feature4by4 = double_blaze_block(db4, filters=96, mid_channels=24, phase_train=phrase_train)
            db5 = double_blaze_block(feature4by4, filters=96, mid_channels=24, stride=2, phase_train=phrase_train)
            db5 = double_blaze_block(db5, filters=96, mid_channels=24, phase_train=phrase_train)
            db5 = double_blaze_block(db5, filters=96, mid_channels=24, phase_train=phrase_train)
            feature2by2 = double_blaze_block(db5, filters=96, mid_channels=24, phase_train=phrase_train)
            # db6 = double_blaze_block(feature2by2, filters=96, mid_channels=24, stride=2, phase_train=phrase_train)
            # db6 = double_blaze_block(db6, filters=96, mid_channels=24, phase_train=phrase_train)
            # db6 = double_blaze_block(db6, filters=96, mid_channels=24, phase_train=phrase_train)
            # feature1by1 = double_blaze_block(db6, filters=96, mid_channels=24, stride=2, phase_train=phrase_train)

    feature1 = tf.identity(feature32by32, 'feature_map1')
    feature2 = tf.identity(feature16by16, 'feature_map2')
    feature3 = tf.identity(feature8by8, 'feature_map3')
    feature4 = tf.identity(feature4by4, 'feature_map4')
    feature5 = tf.identity(feature2by2, 'feature_map5')
    # feature6 = tf.identity(feature1by1, 'feature_map6')
    return inputs, phrase_train, feature1, feature2, feature3, feature4, feature5, #feature6


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


def build_prediction_convs(input_shape=(128, 128)):
    inputs, phrase_train, feature1, feature2, feature3, feature4, feature5 = build_backbone(input_shape)
    # attach feature map1 to a conv to output coordinate prediction
    # anchor ratio is always 1*1
    predict1 = tf.layers.conv2d(feature1, filters=5, strides=2, kernel_size=2, padding='SAME')
    predict2 = tf.layers.conv2d(feature2, filters=5, strides=2, kernel_size=2, padding='SAME')
    predict3 = tf.layers.conv2d(feature3, filters=5, strides=2, kernel_size=2, padding='SAME')
    predict4 = tf.layers.conv2d(feature4, filters=5, strides=2, kernel_size=2, padding='SAME')
    predict5 = tf.layers.conv2d(feature5, filters=5, strides=2, kernel_size=2, padding='SAME')
    # predict6 = tf.layers.conv2d(feature6, filters=5, strides=1, kernel_size=1, padding='SAME')
    return inputs, phrase_train, tf.identity(predict1, 'predict1'), tf.identity(predict2, 'predict2'), predict3, \
           predict4, predict5


if __name__ == '__main__':
    # test output feature's shape
    shape = (512, 512)
    inputs, phrase_train, predict1, predict2, predict3, predict4, predict5 = build_prediction_convs(shape)
    # run_meta = tf.RunMetadata()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out1, out2, out3, out4, out5 = sess.run((predict1, predict2, predict3, predict4, predict5),
                                                      feed_dict={inputs: np.random.random((16, *shape, 3)),
                                                                 phrase_train: True})
        print(out1.shape)
        print(out2.shape)
        print(out3.shape)
        print(out4.shape)
        print(out5.shape)
