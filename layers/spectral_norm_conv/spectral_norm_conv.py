import tensorflow as tf


def _spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.Variable(tf.random_normal_initializer()(
        [1, w_shape[-1]]), name="u", trainable=False)

    u_hat = u
    v_hat = None
    for _ in range(iteration):

        """
        power iteration
        Usually iteration = 1 will be enough
        """

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


class SpectralNormConv2d(tf.keras.layers.Layer):
    def __init__(self, kernel_size, out_channels, stride, activation=None):
        super(SpectralNormConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.activation = activation
    
    def build(self, input_shape):
        self.w = tf.Variable(tf.initializers.glorot_uniform()(
            [self.kernel_size, self.kernel_size, input_shape[-1], self.out_channels]), trainable=True, name="kernel")
        self.b = tf.Variable(tf.constant_initializer(0)(
            [self.out_channels]), trainable=True, name="bias")

    def call(self, x):
        if self.activation:
            return self.activation(tf.nn.conv2d(input=x, filter=_spectral_norm(self.w), strides=[
                self.stride, self.stride], padding='SAME') + self.b)
        else:
            return tf.nn.conv2d(input=x, filter=_spectral_norm(self.w), strides=[
                self.stride, self.stride], padding='SAME') + self.b
