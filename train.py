import tensorflow as tf
from layers.atnconv.atnconv import AtnConv


def main():
    input_placeholder, loss = build_network()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        for i in range(train_steps):
            # TODO: Load examples here.
            _, loss_val = sess.run([train_op, loss], feed_dict={input_placeholder: images})
            print('loss: {}'.format(loss_val))

def preprocess(input_img):
    return input_img/255.0

def build_network():
    input_img = tf.placeholder(tf.float32, shape=[256, 256, 4])
    processed_input = preprocess(input_img)
    cnum = 32
    x = tf.conv2d(input_image, cnum//2, ksize=3, stride=1)

    # encode
    enc_feats = []
    dims = [cnum * i for i in [1, 2, 4, 8, 8, 8]]
    activations = [tf.nn.leaky_relu] * 6 + [tf.nn.relu]
    for i in range(len(dims)):
        enc_feats.append(x)
        x = tf.conv2d(x, dims[i], ksize=3, stride=2, activation=activations[i])
    latent_feat = x

    # attention transfer networks
    attn_feats = []
    x = latent_feat
    for i in range(len(dims)):
        x = AtnConv(enc_feats[-(i+1)], x, mask)
        attn_feats.append(x)

    # decode
    x = latent_feat
    dims = [cnum * i for i in [1./2, 1, 2, 4, 8, 8]]
    outputs = [None] * range(len(dims))
    for i in range(len(dims)):
        x = tf.deconv2d(x, dims[-(i+1)], ksize=3, stride=2, activation=tf.nn.relu)
        x = tf.concat([x, attn_feats[i]], axis=3)
        outputs[-(i+1)] = tf.clip_by_value(tf.conv2d(x, 3, ksize=1, stride=1, activation=activations[i]), 0, 1)

    l1_loss = tf.get_variable('l1_loss', shape=[], dtype=tf.float32, initializer=tf.zero_initializer)
    raw_img = tf.slice(input_image, [0, 0, 0, 0], [-1, -1, -1, 3])
    for output in outputs:
        output_shape = tf.shape(output)
        l1_loss += tf.losses.absolute_difference(
                tf.resize_images(
                    raw_img,
                    size=[output_shape[1], output_shape[2]]),
                output_shape)

    return input_image, l1_loss

if __name__ == "__main__":
    main()
