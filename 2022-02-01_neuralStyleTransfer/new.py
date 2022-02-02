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


"""
content_path = tf.keras.utils.get_file('notebook-styletransfer-bigben.jpg',
                                       "https://c8.alamy.com/compde/e7rb1g/big-ben-in-london-e7rb1g.jpg")
"""
content_path = tf.keras.utils.get_file('jean.profile',
                                       "file:///home/jean/Pictures/profile_hippie.jpeg")


style_path = tf.keras.utils.get_file('starry-night.png',
                                     "https://images.fineartamerica.com/images/artworkimages/mediumlarge/3/23-starry-night-vincent-van-gogh.jpg")

content_image = load_img(content_path)
style_image = load_img(style_path)


"""
plt.subplot(1, 2, 2)
imgshow(content_image, 'Content image')
plt.show()
"""


import tensorflow_hub as hub
hub_model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

plt.subplot(1, 2, 2)
imgshow(stylized_image, 'styled image')
plt.show()
"""

tensor_to_image(style_image)


x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
prediction_propababilites = vgg(x)
print(prediction_propababilites.shape)

predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_propababilites.numpy())[0]

[(class_name, prob) for (number, class_name, prob) in predicted_top_5]

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')


for layer in vgg.layers:
    print(layer.name)
"""