import pathlib
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from layers.atnconv import AtnConv
from layers.spectral_norm_conv import SpectralNormConv2d

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


def load_dataset(data_dir, batch_size):
    def preprocess_image(image):
        image = tf.image.decode_image(
            image, channels=3, dtype=tf.float32, expand_animations=False)
        image = tf.image.resize(image, [IMAGE_DIM, IMAGE_DIM])
        image.set_shape([IMAGE_DIM, IMAGE_DIM, 3])
        # image /= 255.0  # normalize to [0,1] range

        return image

    def load_and_preprocess_image(path):
        image = tf.compat.v1.read_file(path)
        return preprocess_image(image)

    all_image_paths = tf.data.Dataset.list_files(data_dir+'*')
    image_ds = all_image_paths.map(load_and_preprocess_image, num_parallel_calls=8)
    image_ds = image_ds.batch(batch_size).prefetch(2)

    return image_ds


def preprocess(input_img):
    mask_val = np.zeros(
        shape=[BATCH_SIZE, IMAGE_DIM, IMAGE_DIM, 1], dtype=np.float32)
    mask_val[:, ((IMAGE_DIM - PATCH_DIM)//2):((IMAGE_DIM + PATCH_DIM)//2),
             ((IMAGE_DIM - PATCH_DIM)//2):((IMAGE_DIM + PATCH_DIM)//2), 0] = 1
    mask = tf.constant(mask_val, dtype=tf.float32, name='mask')
    masked_img = input_img * (-mask + 1)
    return masked_img, mask


class Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        dims = [64, 128, 256, 512, 1]
        strides = [2, 2, 2, 2, 1]
        activations = [tf.nn.leaky_relu] * 4 + [None]
        self.sn_conv2d = []
        for dim, stride, activation in zip(dims, strides, activations):
            self.sn_conv2d.append = SpectralNormConv2d(5, dim, stride, activation)
        self.dense = tf.keras.layers.Dense(1)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, input_img):
        x = input_img
        for layer in self.sn_conv2d:
            x = layer(x)
        return self.dense(self.flatten(x))


class Generator(tf.keras.layers.Layer):
    def __init__(self):
        super(Generator, self).__init__()
        cnum = 32
        self.conv_0 = tf.keras.layers.Conv2D(filters=cnum//2,
                                             kernel_size=3, strides=1, padding='SAME', activation=tf.nn.leaky_relu)

        # encode
        dims = [cnum * i for i in [1, 2, 4, 8, 8, 8]]
        activations = [tf.nn.leaky_relu] * 6 + [tf.nn.relu]
        self.encode_conv = []
        for i in range(len(dims)):
            self.encode_conv.append(tf.keras.layers.Conv2D(
                filters=dims[i], kernel_size=3, strides=2, padding='SAME', activation=activations[i]))

        # attention transfer networks
        self.attn_conv = []
        for i in range(len(dims)):
            if i > 0:
                out_channels = dims[i-1]
            else:
                out_channels = cnum//2
            self.attn_conv.append(AtnConv(out_channels))

        # decode
        dims = [cnum * i for i in [1./2, 1, 2, 4, 8, 8]]
        self.decode_conv = []
        for i in range(len(dims)):
            self.decode_conv.append(tf.keras.layers.Conv2DTranspose(filters=int(
                dims[-(i+1)]), kernel_size=3, strides=2, padding='SAME', activation=tf.nn.relu)
            )

        # output
        self.output_conv = []
        for i in range(len(dims)):
            self.output_conv.append(tf.keras.layers.Conv2D(
                filters=3, kernel_size=1, strides=1, padding='SAME'))

    def call(self, input_img, mask):
        x = tf.concat([input_img, mask], axis=3)
        x = self.conv_0(x)

        # encode
        enc_feats = []
        for layer in self.encode_conv:
            enc_feats.append(x)
            x = layer(x)
        latent_feat = x

        # attention transfer networks
        attn_feats = []
        x = latent_feat
        for idx, layer in enumerate(self.attn_conv):
            x = layer(enc_feats[-(idx+1)], x,
                      tf.expand_dims(mask[0, :, :, :], 0))
            attn_feats.append(x)

        # decode & output
        x = latent_feat
        outputs = [None] * len(self.decode_conv)
        for idx, (decode_layer, attn_feat, output_layer) in enumerate(zip(self.decode_conv, attn_feats, self.output_conv)):
            x = decode_layer(x)
            x = tf.concat([x, attn_feat], axis=3)
            outputs[-(idx+1)] = tf.clip_by_value(output_layer(x), 0, 1)

        final_output = input_img + outputs[0] * mask
        return outputs, final_output


class LandmarkModel(tf.keras.Model):
    def __init__(self):
        super(LandmarkModel, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def call(self, input_img):
        processed_input, mask = preprocess(input_img)
        layer_outputs, final_output = self.generator(processed_input, mask)
        real_logits = self.discriminator(input_img)
        fake_logits = self.discriminator(final_output)
        d_loss = tf.maximum(.0, 1 - real_logits) + \
            tf.maximum(.0, 1 + fake_logits)

        l1_losses = []
        for output in layer_outputs:
            output_shape = tf.shape(output)
            l1_losses.append(tf.compat.v1.losses.absolute_difference(
                tf.compat.v1.image.resize_images(
                    input_img,
                    size=[output_shape[1], output_shape[2]]),
                output))
        g_loss = tf.add_n(l1_losses) - fake_logits
        return d_loss, g_loss, final_output

@tf.function
def train(train_dataset, model, optimizer, d_loss_metric, g_loss_metric):
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape(persistent=True) as tape:
            d_loss, g_loss, _ = model(x_batch_train)

        d_grads = tape.gradient(
            d_loss, model.discriminator.trainable_weights)
        g_grads = tape.gradient(
            g_loss, model.generator.trainable_weights)
        del tape
        optimizer.apply_gradients(
            zip(d_grads, model.discriminator.trainable_weights))
        optimizer.apply_gradients(
            zip(g_grads, model.generator.trainable_weights))

        d_loss_metric(d_loss)
        g_loss_metric(g_loss)

        print('step %s: mean d_loss = %s, mean g_loss = %s' %
                (step, d_loss_metric.result(), g_loss_metric.result()))

def main():
    train_dataset = load_dataset(DATA_DIR_PATH, BATCH_SIZE)
    lmk_model = LandmarkModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    d_loss_metric = tf.keras.metrics.Mean()
    g_loss_metric = tf.keras.metrics.Mean()

    epochs = 3
    # Iterate over epochs.
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))

        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_dataset):
            train(train_dataset, lmk_model, optimizer, d_loss_metric, g_loss_metric)


if __name__ == '__main__':
    main()
