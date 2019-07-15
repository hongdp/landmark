import tensorflow as tf
import pathlib
import numpy as np
from layers.atnconv.atnconv import AtnConv

DATA_DIR_PATH = '/mnt/shared_data/Workspace/landmark/img/2449/'
BATCH_SIZE = 2
TRAIN_STEPS = 1000
IMAGE_DIM = 128

def main():
    dataset_batch = load_dataset()
    input_placeholder, loss = build_network(dataset_batch)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAIN_STEPS):
            _, loss_val = sess.run([train_op, loss])
            print('loss: {}'.format(loss_val))

def load_dataset():
    def preprocess_image(image):
        image = tf.image.decode_image(image, channels=3, dtype=tf.float32, expand_animations=False)
        image = tf.image.resize(image, [IMAGE_DIM, IMAGE_DIM])
        image.set_shape([IMAGE_DIM, IMAGE_DIM, 3])
        image /= 255.0  # normalize to [0,1] range

        return image

    def load_and_preprocess_image(path):
        image = tf.read_file(path)
        return preprocess_image(image)

    data_root = pathlib.Path(DATA_DIR_PATH)
    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=8)
    image_ds = image_ds.repeat().batch(BATCH_SIZE)

    iter = image_ds.make_one_shot_iterator()
    el = iter.get_next()
    return el

def preprocess(input_img):
    mask_val = np.zeros(shape=[BATCH_SIZE, IMAGE_DIM, IMAGE_DIM, 1], dtype=np.float32)
    mask_val[:, 64:196, 64:196, 0] = 1
    mask = tf.constant(mask_val, dtype=tf.float32, name='mask')
    masked_img = input_img * mask
    final_img = tf.concat([masked_img, mask], axis=3)
    return final_img, mask

def build_network(input_img):
    # input_img = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_DIM, IMAGE_DIM, 3])
    processed_input, mask = preprocess(input_img)
    cnum = 32
    x = tf.layers.conv2d(processed_input, filters=cnum//2, kernel_size=3, strides=1, padding='SAME')

    # encode
    enc_feats = []
    dims = [cnum * i for i in [1, 2, 4, 8, 8, 8]]
    activations = [tf.nn.leaky_relu] * 6 + [tf.nn.relu]
    for i in range(len(dims)):
        enc_feats.append(x)
        x = tf.layers.conv2d(x, filters=dims[i], kernel_size=3, strides=2, padding='SAME', activation=activations[i])
    latent_feat = x

    # attention transfer networks
    attn_feats = []
    x = latent_feat
    for i in range(len(dims)):
        x = AtnConv(enc_feats[-(i+1)], x, tf.expand_dims(mask[0,:,:,:], 0))
        attn_feats.append(x)

    # decode
    x = latent_feat
    dims = [cnum * i for i in [1./2, 1, 2, 4, 8, 8]]
    outputs = [None] * len(dims)
    for i in range(len(dims)):
        attn_feats_shape = tf.shape(attn_feats[i])
        x = tf.layers.conv2d_transpose(x, filters=int(dims[-(i+1)]), kernel_size=3, strides=2, padding='SAME', activation=tf.nn.relu)
        x = tf.concat([x, attn_feats[i]], axis=3)
        outputs[-(i+1)] = tf.clip_by_value(tf.layers.conv2d(x, filters=3, kernel_size=3, strides=1, padding='SAME', activation=activations[i]), 0, 1)

    l1_losses = []
    raw_img = tf.slice(processed_input, [0, 0, 0, 0], [-1, -1, -1, 3])
    for output in outputs:
        output_shape = tf.shape(output)
        l1_losses.append(tf.losses.absolute_difference(
                tf.image.resize_images(
                    raw_img,
                    size=[output_shape[1], output_shape[2]]),
                output))
    loss = tf.add_n(l1_losses)

    return input_img, loss

if __name__ == "__main__":
    main()
