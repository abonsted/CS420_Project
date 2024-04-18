import os
import numpy as np
from PIL import Image

IMG_SIZE = (32, 32)

def image_to_array_BW(image):
    img = Image.open(image)
    img = img.resize(IMG_SIZE)
    grey = img.convert('L')
    return np.where(np.array(grey) < 128, 0, 255)

def image_to_array_BIN(image):
    img = Image.open(image)
    img = img.resize(IMG_SIZE)
    bin = img.convert('1')
    return np.array(bin)

def array_to_image_BW(array):
    return Image.fromarray(np.uint8(array))

def array_to_image_BIN(array):
    return Image.fromarray(np.bool_(array))

def image_to_hopfield(image, type='BW'):
    if type == 'BW':
        arr = image_to_array_BW(image)
    elif type == 'BIN':
        arr = image_to_array_BIN(image)

    return np.where(arr == 255, 1, -1).ravel()

def hopfield_to_image(arr, type='BW'):
    arr = np.where(arr == 1, 255, 0)
    arr.resize(IMG_SIZE)

    if type == 'BW':
        image = array_to_image_BW(arr)
    elif type == 'BIN':
        image = array_to_image_BIN(arr)

    return image

def load_hopfield(path):
    images = []
    files = os.listdir(path)

    for i in range(len(files)):
        file = f'{path}/{files[i]}'
        image = image_to_hopfield(file)
        images.append(image)

    return np.array(images)


if __name__ == '__main__':
    images = load_hopfield('images/test')