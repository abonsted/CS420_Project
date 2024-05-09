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

def image_to_hopfield(image, type):
    if type == 'BW':
        arr = image_to_array_BW(image)
    elif type == 'BIN':
        arr = image_to_array_BIN(image)

    return np.where(arr == 255, 1, -1).ravel()

def hopfield_to_image(arr, type):
    arr = np.where(arr == 1, 255, 0)
    arr.resize(IMG_SIZE)

    if type == 'BW':
        image = array_to_image_BW(arr)
    elif type == 'BIN':
        image = array_to_image_BIN(arr)

    return image


if __name__ == '__main__':
    bin_array = image_to_array_BW('test.jpg')
    bin_image = array_to_image_BW(bin_array)
    bin_image.save('bin_image.jpg')