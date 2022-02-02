import os
import tensorflow as tf


import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False


import numpy as np
import PIL.Image
import time
import functools

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis,:]
    return img


def imgshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)

    if title:
        plt.title(title)


content_path = tf.keras.utils.get_file('notebook-styletransfer-bigben.jpg',
                                       "https://c8.alamy.com/compde/e7rb1g/big-ben-in-london-e7rb1g.jpg")

style_path = tf.keras.utils.get_file('starry-night.png',
                                     "https://images.fineartamerica.com/images/artworkimages/mediumlarge/3/23-starry-night-vincent-van-gogh.jpg")

content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)

imgshow(content_image, 'Content image')

plt.subplot(1, 2, 2)
imgshow(style_image, 'Style image')

plt.show()
