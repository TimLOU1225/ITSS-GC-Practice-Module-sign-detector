from __future__ import division
import numpy as np
import scipy.io
import scipy.ndimage
import cv2
import imageio
from PIL import Image

def augment(img, saliencymap, rot_90=True):
    if rot_90:
        angle = np.random.choice([0, 90, 180, 270], 1)[0]
        if angle == 270:
            img = cv2.flip(img, 0)
            saliencymap = cv2.flip(saliencymap, 0)
        elif angle == 180:
            img = cv2.flip(img, -1)
            saliencymap = cv2.flip(saliencymap, -1)
        elif angle == 90:
            img = cv2.flip(img, 1)
            saliencymap = cv2.flip(saliencymap, 1)
        elif angle == 0:
            pass
    return img, saliencymap

def padding(img, shape_r=240, shape_c=320, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    img = Image.fromarray(img)
    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = img.resize((new_cols, shape_r), Image.Resampling.LANCZOS)
        img = np.array(img)
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols),] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = img.resize((shape_c, new_rows), Image.Resampling.LANCZOS)
        img = np.array(img)
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

def preprocess_images(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))

    for i, path in enumerate(paths):
        original_image = imageio.imread(path)
        if original_image.ndim == 2:
            copy = np.zeros((original_image.shape[0], original_image.shape[1], 3))
            copy[:, :, 0] = original_image
            copy[:, :, 1] = original_image
            copy[:, :, 2] = original_image
            original_image = copy
        padded_image = padding(original_image, shape_r, shape_c, 3)
        ims[i] = padded_image

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68
    ims = ims[:, :, :, ::-1]
    return ims

def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1))

    for i, path in enumerate(paths):
        original_map = imageio.imread(path)
        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i, :, :, 0] = padded_map.astype(np.float32)
        ims[i, :, :, 0] /= 255.0

    return ims

def postprocess_predictions(pred, shape_r, shape_c):
    pred = Image.fromarray(pred)
    pred = pred.resize((shape_r, shape_c), Image.Resampling.LANCZOS)
    pred = np.array(pred)
    pred = pred / np.max(pred) * 255
    return pred
