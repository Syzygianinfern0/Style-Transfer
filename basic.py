import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import sleep
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def read_images(path1, path2):
    img1 = cv2.imread(path1, 1)
    img2 = cv2.imread(path2, 1)
    img1 = cv2.resize(img1, (512, 512))
    img2 = cv2.resize(img2, (512, 512))
    return img1, img2


def preprocess(image):
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image


def gram_matrix(input_tensor: tf.Tensor):
    # b - Batch size
    # i - Rows
    # j - Cols
    # c (d) - Channels (letters cannot be reused so we use d)
    # We need to reduce_sum across the i'th dimension and multiply the transpose and itself in the next 2 dimensions
    # it is equivalent to : np.dot(a[0][0].T,a[0][0]) + np.dot(a[0][1].T, a[0][1])
    # if 'a' is of shape (1,2,3,4)
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    return result / tf.shape(input_tensor.shape[1] * input_tensor.shape[2])  # Normalize


# noinspection PyShadowingNames
def get_model(content, style):
    # Loading the Awesome VGG. Set weights to a constant to avoid changing them
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    layers = content + style
    out = [vgg.get_layer(layer).output for layer in layers]
    return tf.keras.Model(vgg.input, out)


if __name__ == '__main__':
    content = ['block5_conv2']  # From the research paper, these were the best layers to work with
    style = ['block1_conv1',
             'block2_conv1',
             'block3_conv1',
             'block4_conv1',
             'block5_conv1']

    content_img, style_img = read_images('assets/content_images/nitt.jpg', 'assets/style_images/starry.jpg')
