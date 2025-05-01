from io import BytesIO
import os
import cv2
import base64
import numpy as np
import tensorflow as tf
from PIL import Image


def crop_image(image, mask, target_size=(384, 384)):
    img_uint8 = tf.cast(image, tf.uint8)

    resize_image = tf.image.resize(img_uint8, (512, 1024), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    resize_mask = tf.image.resize(mask, (512, 1024), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    start_y = (512 - target_size[0]) // 2
    start_x = (1024 - target_size[1]) // 2

    cropped_img = tf.image.crop_to_bounding_box(resize_image, start_y, start_x, target_size[0], target_size[1])
    cropped_mask = tf.image.crop_to_bounding_box(resize_mask, start_y, start_x, target_size[0], target_size[1])

    return cropped_img, tf.cast(cropped_mask, mask.dtype)


def load_image(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)

    with tf.device('/cpu:0'):
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.where(mask == 255, tf.zeros_like(mask), mask)

    img, mask = crop_image(img, mask)

    img = tf.cast(img, tf.float32) / 127.5 - 1
    mask = tf.cast(mask, tf.int32)

    img = tf.expand_dims(img, axis=0)
    return img, mask


def load_image_without_mask(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 127.5 - 1
    img = tf.expand_dims(img, axis=0)
    return img


def load_image_without_path(img_data):
    if isinstance(img_data, BytesIO):
        img_data = img_data.read()
    img = tf.image.decode_png(img_data, channels=3)
    img = tf.cast(img, tf.float32) / 127.5 - 1
    img = tf.expand_dims(img, axis=0)
    return img


def load_image_for_other_scene(img_path, mask_path):
    IMG_HEIGHT = 384
    IMG_WIDTH = 384

    img = np.array(Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT)))
    seg = np.array(Image.open(mask_path).resize((IMG_WIDTH, IMG_HEIGHT)))
    seg[seg == 255] = 21
    seg = np.expand_dims(seg, axis=-1)
    # image_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    return img, seg

    # # 假设你已经知道你要处理的图片的名称
    # image_name = "example_image"
    # path = "pascal-voc-2012"
    # image, mask = read_single_image(path, image_name)
    #
    # image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    # mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)
    #
    # image_tensor /= 255.0

