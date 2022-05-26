import numpy as np
from imageio import imread
import cv2
from PIL import Image
from torchvision import transforms


def image_read(fname, factor=70):
    """
    :param fname: image path
    :param factor: factor = 70 for radar, factor = 35 for wind, factor = 10 for prep
    """
    img = np.array(imread(fname)/255*factor)

    return img


if __name__ == '__main__':
    image_path = '/home/bing/PycharmProjects/dataSets/weather/TestA/Radar/radar_31218.png'
    image = image_read(image_path)
