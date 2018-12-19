import tensorflow as tf

class Pix2Pix():
    def __init__(self, hierarchy, filter_size, kernel_size, conv_strides, batch_size, image_size):
        self.hierarchy = hierarchy
        self.d_filter = filter_size
        self.g_filter = self.d_filter * (2 ** (self.hierarchy - 2))
        self.kernel_size = kernel_size
        self.conv_strides = conv_strides
        self.batch_size = batch_size
        self.image_size = image_size

    def discriminator(self, defocus, focus, is_training):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            # TODO: change to patchGAN
            pairs = tf.concat([defocus, focus], axis = -1)
            feature = encoder(pairs, self.d_filter, self.kernel_size, self.conv_strides, is_training, self.hierarchy, [])
            output = tf.layers.dense(tf.layers.flatten(feature), 1)
            return output

    def generator(self, defocus, is_training):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            # encoder-decoder + skip connection
            skips = []
            latent = encoder(defocus,self.d_filter, self.kernel_size, self.conv_strides, is_training, self.hierarchy, skips)

            gen = decoder(latent, self.g_filter, 3, self.kernel_size, self.conv_strides, is_training, self.hierarchy, skips)
            return (tf.nn.tanh(gen)+1)/2

    def gradient_penalty(self, defocus, fake_focus, true_focus):
        with tf.variable_scope('gradient_penalty', reuse=tf.AUTO_REUSE):
            # uniform sample
            alpha = tf.random_uniform([self.batch_size, 1], minval=0., maxval=1.)
            alpha = tf.tile(tf.reshape(alpha, [-1, 1, 1, 1]),
                            [1, self.image_size[0], self.image_size[1], 3])

            diff = true_focus - fake_focus
            sample = fake_focus + alpha * diff
            sample_score = self.discriminator(defocus, sample, False)

            # compute gradient
            gradients = tf.gradients(sample_score, sample)
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
            penalty = tf.reduce_mean((slopes - 1) ** 2)
            return penalty

def encoder(x, filter_size, kernel_size, conv_strides, is_training, hierarchy, skips):
    # downsample by strided conv
    assert conv_strides > 1
    for i in range(hierarchy):
        x = conv_factory(x, filter_size, kernel_size, conv_strides, is_training)
        filter_size *= 2
        skips.append(x)
    return x

def decoder(x, filter_size, output_size, kernel_size, conv_strides, is_training, hierarchy, skips):
    # upsampling by strided transpose conv
    assert conv_strides > 1
    for i in range(hierarchy-1, 0, -1):
        x = tf.concat([x,skips[i]], axis=-1)
        x = deconv_factory(x, filter_size, kernel_size, conv_strides, is_training)
        filter_size = int(filter_size / 2)
    x = tf.concat([x, skips[0]], axis=-1)
    x = deconv_factory(x, output_size, kernel_size, conv_strides, is_training, pure=True)
    return x

def conv_factory(x, filter_size, kernel_size, conv_strides, is_training, pure = False):
    conv = tf.layers.conv2d(x, filters=filter_size, kernel_size=kernel_size,
                          strides=[conv_strides, conv_strides], padding='SAME', activation=None)
    if pure:
        return conv
    else:
        bn = tf.layers.batch_normalization(conv, training=is_training)
        relu = tf.nn.relu(bn)
        return relu

def deconv_factory(x, filter_size, kernel_size, conv_strides, is_training, pure = False):
    deconv = tf.layers.conv2d_transpose(x, filters=filter_size, kernel_size=kernel_size,
                          strides=[conv_strides, conv_strides], padding='SAME', activation=None)
    if pure:
        return deconv
    else:
        bn = tf.layers.batch_normalization(deconv, training=is_training)
        relu = tf.nn.relu(bn)
        return relu
