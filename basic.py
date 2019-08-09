import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import sleep


def read_images(path1, path2):
    img1 = cv2.imread(path1, 1)
    img2 = cv2.imread(path2, 1)
    img1 = cv2.resize(img1, (512, 512))
    img2 = cv2.resize(img2, (512, 512))
    return img1, img2


if __name__ == '__main__':
    content_img, style_img = read_images('assets/content_images/nitt.jpg', 'assets/style_images/starry.jpg')

    # cv2.imshow('content_img', content_img)
    # cv2.imshow('style_img', style_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')     # The VGG19 Model
    img = tf.keras.applications.vgg19.preprocess_input(content_img)             # Normalization
    img = tf.image.resize(img, (224, 224))       # Input for VGG19 must be (None,224,224,3)
    prediction_probabilities = vgg(tf.expand_dims(img, axis=0))
    predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
    print([(class_name, prob) for (number, class_name, prob) in predicted_top_5])
