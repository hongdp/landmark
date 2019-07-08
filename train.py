import tensorflow as tf
from layers.atnconv import AtnConv


def main():
    input_img = tf.placeholder()
    output_img = build_network(input_img)


def build_network(input_placeholder):

    cnum = 32
    x = conv2d(input, cnum//2, ksize=3, stride=1)

    # encode
    enc_feats = []
    dims = [cnum * i for i in [1, 2, 4, 8, 8, 8]]
    for i in range(len(dims)):
        enc_feats.append(x)
        x = conv2d(x, dims[i], ksize=3, stride=2)
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
    for i in range(len(dims)):
        x = deconv2d(x, dims[-(i+1)], ksize=3, stride=2)
        x = tf.concat([x, attn_feats[i]], axis=3)

    output = conv2d(x, 3, ksize=1, stride=1)

    return output

if __name__ == "__main__":
    main()
