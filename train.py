from datetime import datetime
from layers.atnconv import AtnConv
import numpy as np
import pathlib
import sys
import tensorflow as tf
from tqdm import tqdm

sys.path.append("/home/hongdp/Workspace/landmark")

DATA_DIR_PATH = '/home/hongdp/Workspace/landmark/local_data/img/'
BATCH_SIZE = 2
TRAIN_STEPS = 1500000
CKPT_INTERVAL = 5000
IMAGE_DIM = 256
PATCH_DIM = IMAGE_DIM // 2
# RESTORE_PATH = "./model/model_0/model.ckpt-1165001"
RESTORE_PATH = None
RUN = 'model_0'


def load_dataset():
    def preprocess_image(image):
        image = tf.image.decode_image(
            image, channels=3, dtype=tf.float32, expand_animations=False)
        image = tf.image.resize(image, [IMAGE_DIM, IMAGE_DIM])
        image.set_shape([IMAGE_DIM, IMAGE_DIM, 3])
        # image /= 255.0  # normalize to [0,1] range

        return image

    def load_and_preprocess_image(path):
        image = tf.read_file(path)
        return preprocess_image(image)

    data_root = pathlib.Path(DATA_DIR_PATH)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=12)
    image_ds = image_ds.repeat().batch(BATCH_SIZE).prefetch(2)

    iter = image_ds.make_one_shot_iterator()
    el = iter.get_next()
    return el


def preprocess(input_img):
    mask_val = np.zeros(
        shape=[BATCH_SIZE, IMAGE_DIM, IMAGE_DIM, 1], dtype=np.float32)
    mask_val[:, ((IMAGE_DIM - PATCH_DIM)//2):((IMAGE_DIM + PATCH_DIM)//2),
             ((IMAGE_DIM - PATCH_DIM)//2):((IMAGE_DIM + PATCH_DIM)//2), 0] = 1
    mask = tf.constant(mask_val, dtype=tf.float32, name='mask')
    masked_img = input_img * (-mask + 1)
    return masked_img, mask


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable(
        "u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):

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


def build_discriminator(input_img, reuse=None):

    with tf.variable_scope('discriminator', reuse=reuse):
        dims = [64, 128, 256, 512, 1]
        strides = [2, 2, 2, 2, 1]
        activation = [tf.nn.leaky_relu] * 4 + [None]
        x = input_img
        for i in range(len(dims)):
            with tf.variable_scope('snconv_%d' % (i), reuse=reuse):
                w = tf.get_variable(
                    "kernel", shape=[5, 5, x.get_shape()[-1], dims[i]])
                b = tf.get_variable(
                    "bias", [dims[i]], initializer=tf.constant_initializer(0.0))
                x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[
                                 1, strides[i], strides[i], 1], padding='SAME') + b
                if activation[i]:
                    x = activation[i](x)

        logits = tf.layers.dense(tf.layers.flatten(x), 1)
    return logits


def build_generator(input_img, mask, reuse=None):

    with tf.variable_scope('generator', reuse=reuse):
        cnum = 32
        x = tf.concat([input_img, mask], axis=3)
        x = tf.layers.conv2d(x, filters=cnum//2,
                             kernel_size=3, strides=1, padding='SAME', activation=tf.nn.leaky_relu)

        # encode
        enc_feats = []
        dims = [cnum * i for i in [1, 2, 4, 8, 8, 8]]
        activations = [tf.nn.leaky_relu] * 6 + [tf.nn.relu]
        for i in range(len(dims)):
            enc_feats.append(x)
            x = tf.layers.conv2d(
                x, filters=dims[i], kernel_size=3, strides=2, padding='SAME', activation=activations[i])
        latent_feat = x

        # attention transfer networks
        attn_feats = []
        x = latent_feat
        for i in range(len(dims)):
            x = AtnConv(enc_feats[-(i+1)], x,
                        tf.expand_dims(mask[0, :, :, :], 0))
            attn_feats.append(x)

        # decode
        x = latent_feat
        dims = [cnum * i for i in [1./2, 1, 2, 4, 8, 8]]
        outputs = [None] * len(dims)
        for i in range(len(dims)):
            attn_feats_shape = tf.shape(attn_feats[i])
            x = tf.layers.conv2d_transpose(x, filters=int(
                dims[-(i+1)]), kernel_size=3, strides=2, padding='SAME', activation=tf.nn.relu)
            x = tf.concat([x, attn_feats[i]], axis=3)
            outputs[-(i+1)] = tf.clip_by_value(tf.layers.conv2d(x, filters=3,
                                                                kernel_size=1, strides=1, padding='SAME'), 0, 1)
        final_output = input_img + outputs[0] * mask
    return outputs, final_output


def build_network(input_img):
    # input_img = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_DIM, IMAGE_DIM, 3])
    processed_input, mask = preprocess(input_img)

    layer_outputs, final_output = build_generator(processed_input, mask)
    real_logits = build_discriminator(input_img)
    fake_logits = build_discriminator(final_output, reuse=True)
    d_loss = tf.maximum(.0, 1 - real_logits) + tf.maximum(.0, 1 + fake_logits)

    l1_losses = []
    for output in layer_outputs:
        output_shape = tf.shape(output)
        l1_losses.append(tf.losses.absolute_difference(
            tf.image.resize_images(
                input_img,
                size=[output_shape[1], output_shape[2]]),
            output))
    g_loss = tf.add_n(l1_losses) - fake_logits

    logdir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    for l1_layer in range(len(l1_losses)):
        tf.summary.scalar('l1_loss_%d' % l1_layer,
                          tf.reduce_mean(l1_losses[l1_layer]))
        tf.summary.image('output_%d' % l1_layer, layer_outputs[l1_layer])
    tf.summary.scalar('fake_logits',  tf.reduce_mean(fake_logits))
    tf.summary.scalar('real_logits',  tf.reduce_mean(real_logits))
    tf.summary.scalar('g_loss',  tf.reduce_mean(g_loss))
    tf.summary.scalar('d_loss',  tf.reduce_mean(d_loss))
    tf.summary.image('processed_input', processed_input)
    tf.summary.image('final_output', final_output)
    tf.summary.image('input', input_img)

    merged = tf.summary.merge_all()

    return g_loss, d_loss, merged


def main():
    dataset_batch = load_dataset()
    g_loss, d_loss, summary = build_network(dataset_batch)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    # optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
    global_step_tensor = tf.train.get_or_create_global_step()
    train_g_op = optimizer.minimize(g_loss, var_list=tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'), global_step=global_step_tensor)
    train_d_op = optimizer.minimize(d_loss, var_list=tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()

    logdir = "./logs/" + RUN
    train_writer = tf.summary.FileWriter(logdir, sess.graph)

    if RESTORE_PATH:
        saver.restore(sess, RESTORE_PATH)
    else:
        sess.run(tf.global_variables_initializer())

    g_steps = sess.run(global_step_tensor)
    with tqdm(total=TRAIN_STEPS) as pbar:
        pbar.update(g_steps)
        while g_steps < TRAIN_STEPS:
            _, _, summary_val, g_steps = sess.run(
                [train_g_op, train_d_op, summary, global_step_tensor])
            if not g_steps % 100:
                train_writer.add_summary(
                    summary_val, tf.compat.v1.train.global_step(sess, global_step_tensor))
            if not g_steps % CKPT_INTERVAL:
                saver.save(
                    sess, "./model/%s/model.ckpt" % (RUN), global_step=global_step_tensor)
            pbar.update(1)

    saver.save(sess, "./model/%s/model.ckpt" %
               (RUN), global_step=global_step_tensor)


if __name__ == '__main__':
    main()
