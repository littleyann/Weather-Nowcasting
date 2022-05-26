import numpy as np
from imageio.v2 import imread
import pandas as pd
import matplotlib.pyplot as plt


def image_read(fname, factor=70):
    """
    :param fname: image path
    :param factor: factor = 70 for radar, factor = 35 for wind, factor = 10 for prep
    """
    img = np.array(imread(fname)/255*factor)
    return img


def load_csv(path):
    file_data = pd.read_csv(path)
    file_data = np.array(file_data)

    return file_data


if __name__ == '__main__':
    data_line = load_csv('TestA.csv')
    data_pair1 = data_line[0, :]
    print(data_pair1)
    data = np.zeros((40, 480, 560))
    # train_data = np.zeros((20, 480, 560))
    # test_test = np.zeros((20, 480, 560))
    for i, img_name in enumerate(data_pair1):
        img_path = '/home/bing/PycharmProjects/dataSets/weather/TestA/Radar/radar_' + img_name
        data[i, :, :] = image_read(img_path)

    train_data = data[0: 20, :, :]
    test_data = data[20: 40, :, :]


