import tensorflow as tf

class StarGAN():
    def __init__(
        self,
        batch_size=1,
        image_shape=[160,160,3],
        domains=8,
    ):
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.domains = domains

    def generate(self, Z_image, Z_domain):
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            def residual_block(x, scope, chs=64):
                with tf.variable_scope(scope):
                    y = tf.contrib.layers.instance_norm(x)
                    y = tf.nn.relu(y)
                    y = tf.layers.conv2d(x, chs, [3, 3], padding="same")
                    y = tf.contrib.layers.instance_norm(y)
                    y = tf.nn.relu(y)
                    y = tf.layers.conv2d(y, chs, [3, 3], padding="same")
                net = x + y
                return net
            domain = tf.tile(tf.reshape(Z_domain, [Z_image.shape[0], 1, 1, self.domains]), [1, Z_image.shape[1], Z_image.shape[2], 1])
            net = tf.concat([Z_image, domain], axis=-1)
            net = tf.layers.conv2d(net, 64, [7, 7], padding="same")
            net = tf.contrib.layers.instance_norm(net)
            net = tf.nn.relu(net)
            net = tf.layers.conv2d(net, 128, [4, 4], [2, 2], padding="same")
            net = tf.contrib.layers.instance_norm(net)
            net = tf.nn.relu(net)
            net = tf.layers.conv2d(net, 256, [4, 4], [2, 2], padding="same")
            net = tf.contrib.layers.instance_norm(net)
            net = tf.nn.relu(net)
            net = tf.contrib.layers.repeat(net, 8, residual_block, chs=256, scope="residual_blocks")
            net = tf.layers.conv2d_transpose(net, 128, [4, 4], [2, 2], padding="same")
            net = tf.contrib.layers.instance_norm(net)
            net = tf.nn.relu(net)
            net = tf.layers.conv2d_transpose(net, 64, [4, 4], [2, 2], padding="same")
            net = tf.contrib.layers.instance_norm(net)
            net = tf.nn.relu(net)
            net = tf.cast(net, tf.float32)
            net = tf.layers.conv2d(net, 3, [7, 7], padding="same")
            net = tf.nn.tanh(net)
        return net

    def discriminate(self, Z):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            def conv2d_batch(x, filters, kernel_size=[4, 4], strides=[1, 1], scope="conv_batch"):
                with tf.variable_scope(scope):
                    x = tf.layers.conv2d(x, filters, kernel_size, strides, padding="same")
                    # x = tf.layers.batch_normalization(x)
                    x = tf.nn.leaky_relu(x)
                return x
            net = tf.layers.conv2d(Z, 64, [4, 4], [2, 2], padding="same")
            net = tf.nn.leaky_relu(net)
            net = conv2d_batch(net, 128, strides=[2, 2], scope="conv2")
            net = conv2d_batch(net, 256, strides=[2, 2], scope="conv4")
            net = conv2d_batch(net, 512, strides=[2, 2], scope="conv6")
            net = conv2d_batch(net, 1024, strides=[2, 2], scope="conv8")

            net_im = conv2d_batch(net, 2048, [3, 3], [1, 1], scope="conv10")
            net_im = tf.layers.flatten(net)
            net_im = tf.layers.dense(net_im, 1)
            net_im = tf.cast(net_im, tf.float32)
            net_im = tf.nn.sigmoid(net_im)

            net_dom = conv2d_batch(net, self.domains, [3, 3], [1, 1], scope="conv11")
            net_dom = tf.reduce_mean(net_dom, axis=[1, 2])
            net_dom = tf.cast(net_dom, tf.float32)
            net_dom = tf.nn.sigmoid(net_dom)
        return net_im, net_dom

