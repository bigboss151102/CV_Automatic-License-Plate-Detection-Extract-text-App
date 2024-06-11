import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt

print(tf.__version__)

model = tf.keras.models.load_model(
    './static/model/object_detection.h5')

pt.pytesseract.tesseract_cmd = r'C:Users/ADMIN/AppData/Local/Tesseract-OCR/tesseract.exe'


def object_detection(path, filename):
    # read image
    image = load_img(path)  # PIL object
    image = np.array(image, dtype=np.uint8)  # 8 bit array (0,255)
    image1 = load_img(path, target_size=(224, 224))
    # data preprocessing
    # convert into array and get the normalized output
    image_arr_224 = img_to_array(image1)/255.0
    h, w, d = image.shape
    test_arr = image_arr_224.reshape(1, 224, 224, 3)
    # make predictions
    coords = model.predict(test_arr)
    # denormalize the values
    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    # draw bounding on top the image
    xmin, xmax, ymin, ymax = coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    print(pt1, pt2)
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    # convert into bgr
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/predict/{}'.format(filename), image_bgr)
    return coords


def save_text(filename, text):
    name, ext = os.path.splitext(filename)
    with open('./static/predict/{}.txt'.format(name), mode='w') as f:
        f.write(text)
    f.close()


def OCR(path, filename):
    img = np.array(load_img(path))
    cods = object_detection(path, filename)
    xmin, xmax, ymin, ymax = cods[0]
    roi = img[ymin:ymax, xmin:xmax]
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    magic_color = apply_brightness_contrast(gray, brightness=40, contrast=70)
    cv2.imwrite('./static/roi/{}'.format(filename), roi_bgr)

    text = pt.image_to_string(magic_color, lang='eng', config='--psm 6')
    print(text)
    save_text(filename, text)
    return text


def apply_brightness_contrast(input_img, brightness=0, contrast=0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
