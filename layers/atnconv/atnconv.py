import tensorflow as tf


def _resize(x, scale=2, to_shape=None, align_corners=True, dynamic=False,
            func=tf.compat.v1.image.resize_bilinear):
    r""" resize feature map
    https://github.com/JiahuiYu/neuralgym/blob/master/neuralgym/ops/layers.py#L114
    """
    if scale == 1:
        return x
    if dynamic:
        xs = tf.cast(tf.shape(x), tf.float32)
        new_xs = [tf.cast(xs[1]*scale, tf.int32),
                  tf.cast(xs[2]*scale, tf.int32)]
    else:
        xs = x.get_shape().as_list()
        new_xs = [int(xs[1]*scale), int(xs[2]*scale)]
        if to_shape is None:
            x = func(x, new_xs, align_corners=align_corners)
        else:
            x = func(x, [to_shape[0], to_shape[1]],
                     align_corners=align_corners)
    return x


class AtnConv(tf.keras.layers.Layer):
    r""" Attention transfer networks implementation in tensorflow

    Attention transfer networks is introduced in publication:
      Learning Pyramid-Context Encoder Networks for High-Quality Image Inpainting, Zeng et al.
      https://arxiv.org/pdf/1904.07475.pdf
      https://github.com/researchmm/PEN-Net-for-Inpainting

    inspired by:
      Generative Image Inpainting with Contextual Attention, Yu et al.
      https://github.com/JiahuiYu/generative_inpainting/blob/master/inpaint_ops.py
      https://arxiv.org/abs/1801.07892
    """

    def __init__(self, out_channels, ksize=3, stride=1, rate=2,
                 softmax_scale=10., rescale=False):
        r"""
        Args:
          ksize: Kernel size for attention transfer networks.
          stride: Stride for extracting patches from feature map.
          rate: Dilation for matching.
          softmax_scale: Scaled softmax for attention.
          training: Indicating if current graph is training or inference.
          rescale: Indicating if input feature maps need to be downsample
        """
        super(AtnConv, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.softmax_scale = softmax_scale
        self.rescale = rescale

        self.conv1 = tf.keras.layers.Conv2D(
            out_channels//4, 3, 1, dilation_rate=1, activation=tf.nn.relu, padding='SAME')
        self.conv2 = tf.keras.layers.Conv2D(
            out_channels//4, 3, 1, dilation_rate=2, activation=tf.nn.relu, padding='SAME')
        self.conv3 = tf.keras.layers.Conv2D(
            out_channels//4, 3, 1, dilation_rate=4, activation=tf.nn.relu, padding='SAME')
        self.conv4 = tf.keras.layers.Conv2D(
            out_channels//4, 3, 1, dilation_rate=8, activation=tf.nn.relu, padding='SAME')

    def call(self, x1, x2, mask=None):
        r"""
        Args:
          x1:  low-level feature map with larger  size [b, h, w, c].
          x2: high-level feature map with smaller size [b, h/2, w/2, c].
          mask: Input mask, 1 for missing regions 0 for known regions.
        Returns:
          tf.Tensor: reconstructed feature map
        """
        # downsample input feature maps if needed due to limited GPU memory
        if self.rescale:
            x1 = _resize(x1, scale=1./2,
                         func=tf.compat.v1.image.resize_nearest_neighbor)
            x2 = _resize(x2, scale=1./2,
                         func=tf.compat.v1.image.resize_nearest_neighbor)
        # get shapes
        raw_x1s = tf.shape(x1)
        int_x1s = x1.get_shape().as_list()
        int_x2s = x2.get_shape().as_list()
        # extract patches from low-level feature maps for reconstruction
        kernel = 2*self.rate
        raw_w = tf.compat.v1.extract_image_patches(
            x1, [1, kernel, kernel, 1], [1, self.rate*self.stride, self.rate*self.stride, 1], [1, 1, 1, 1], padding='SAME')
        raw_w = tf.reshape(raw_w, [int_x1s[0], -1, kernel, kernel, int_x1s[3]])
        # transpose to [b, kernel, kernel, c, hw]
        raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])
        raw_w_groups = tf.split(raw_w, int_x1s[0], axis=0)
        # extract patches from high-level feature maps for matching and attending
        x2_groups = tf.split(x2, int_x2s[0], axis=0)
        w = tf.compat.v1.extract_image_patches(
            x2, [1, self.ksize, self.ksize, 1], [1, self.stride, self.stride, 1], [1, 1, 1, 1], padding='SAME')
        w = tf.reshape(w, [int_x2s[0], -1, self.ksize, self.ksize, int_x2s[3]])
        # transpose to [b, ksize, ksize, c, hw/4]
        w = tf.transpose(w, [0, 2, 3, 4, 1])
        w_groups = tf.split(w, int_x2s[0], axis=0)
        # resize and extract patches from masks
        mask = _resize(mask, to_shape=int_x2s[1:3],
                       func=tf.compat.v1.image.resize_nearest_neighbor)
        m = tf.compat.v1.extract_image_patches(
            mask, [1, self.ksize, self.ksize, 1], [1, self.stride, self.stride, 1], [1, 1, 1, 1], padding='SAME')
        m = tf.reshape(m, [1, -1, self.ksize, self.ksize, 1])
        # transpose to [1, ksize, ksize, 1, hw/4]
        m = tf.transpose(m, [0, 2, 3, 4, 1])
        m = m[0]
        mm = tf.cast(tf.equal(tf.reduce_mean(
            m, axis=[0, 1, 2], keepdims=True), 0.), tf.float32)

        # matching and attending hole and non-hole patches
        y = []
        scale = self.softmax_scale
        for xi, wi, raw_wi in zip(x2_groups, w_groups, raw_w_groups):
            # matching on high-level feature maps
            wi = wi[0]
            wi_normed = wi / \
                tf.maximum(tf.sqrt(tf.reduce_sum(
                    tf.square(wi), axis=[0, 1, 2])), 1e-4)
            yi = tf.nn.conv2d(xi, wi_normed, strides=[
                              1, 1, 1, 1], padding="SAME")
            yi = tf.reshape(yi, [1, int_x2s[1], int_x2s[2],
                                 (int_x2s[1]//self.stride)*(int_x2s[2]//self.stride)])
            # apply softmax to obtain attention score
            yi *= mm  # mask
            yi = tf.nn.softmax(yi*scale, 3)
            yi *= mm  # mask
            # transfer non-hole features into holes according to the atttention score
            wi_center = raw_wi[0]
            yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_x1s[1:]], axis=0),
                                        strides=[1, self.rate*self.stride, self.rate*self.stride, 1]) / 4.
            y.append(yi)
        y = tf.concat(y, axis=0)
        y.set_shape(int_x1s)
        # refine filled feature map after matching and attending
        y1 = self.conv1(y)
        y2 = self.conv2(y)
        y3 = self.conv3(y)
        y4 = self.conv4(y)
        y = tf.concat([y1, y2, y3, y4], axis=3)
        if self.rescale:
            y = _resize(
                y, scale=2., func=tf.compat.v1.image.resize_nearest_neighbor)
        return y
