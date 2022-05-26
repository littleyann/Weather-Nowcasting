import numpy as np
import torch
import pandas as pd
from dataset.utils import image_read
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class WeatherDataset(Dataset):
    def __init__(self, path, model, state):

        if model == 'radar' and state == 'train':
            self.fold_path = '/home/bing/PycharmProjects/dataSets/weather/Train/Radar/radar_'
            self.factor = 70
        elif model == 'radar' and state == 'test':
            self.fold_path = '/home/bing/PycharmProjects/dataSets/weather/TestA/Radar/radar_'
            self.factor = 70
        elif model == 'wind' and state == 'train':
            self.fold_path = '/home/bing/PycharmProjects/dataSets/weather/Train/Wind/wind_'
            self.factor = 35
        elif model == 'wind' and state == 'test':
            self.fold_path = '/home/bing/PycharmProjects/dataSets/weather/TestA/Wind/wind_'
            self.factor = 35
        elif model == 'precip' and state == 'train':
            self.fold_path = '/home/bing/PycharmProjects/dataSets/weather/Train/Precip/precip_'
            self.factor = 10
        else:
            self.fold_path = '/home/bing/PycharmProjects/dataSets/weather/TestA/Precip/precip_'
            self.factor = 10

        file_data = pd.read_csv(path)
        file_data = np.array(file_data)
        self.file_data = file_data

    def __getitem__(self, index):
        data_pair1 = self.file_data[index, :]
        data = np.zeros((40, 480, 560))
        for i, img_name in enumerate(data_pair1):
            img_path = self.fold_path + img_name
            data[i, :, :] = image_read(img_path, self.factor)
        train_data = data[0: 20, :, :]
        test_data = data[20: 40, :, :]

        return train_data, test_data

    def __len__(self):
        return len(self.file_data)


if __name__ == '__main__':
    dataset = WeatherDataset('TestA.csv', model='radar', state='test')
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=16, pin_memory=True)
    for i, (train, test) in enumerate(dataloader):
        print(train[0, 0, :, :])
        print('Train data shape: {}, index {}'.format(train.shape, i))
