import torch
from Model import Model
from torch.utils.data import DataLoader
from train import train
from dataset.dataset import WeatherDataset
import time
import os
from warmup_scheduler import GradualWarmupScheduler
from loss import CharbonnierLoss
import argparse
from logger import load_logger
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The hyper parameters used in this project')
    parser.add_argument('--num_epochs', default=15, type=int, help='Total training epochs')
    parser.add_argument('--batch_size', default=2, type=int, help='Mini batch size')
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='Weight decay for optimizer')
    parser.add_argument('--warm_up_epochs', default=1, type=int, help='Warm up epochs')
    parser.add_argument('--checkpoint_path', default='./trained_model/Model_2022-05-05 19:07:12/Model_epoch_5.pth',
                        type=str, help='Checkpoint path')
    parser.add_argument('--resume', default=False, type=bool, help='If we use checkpoint')
    parser.add_argument('--parallel', default=True, type=bool, help='If we use multi GPU')
    parser.add_argument('--random_seed', default=1234, type=int, help='The random seed for this project')

    parser.add_argument('--train_path', default='./dataset/Train.csv', type=str)
    parser.add_argument('--valid_path', default='./dataset/TestA.csv', type=str)
    parser.add_argument('--gpu', default='0, 1', type=str)
    parser.add_argument('--mode', default='radar', type=str, help='mode of the nowcasting')
    args = parser.parse_args()

    file_dir = 'Model_' + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    path = os.path.join('/home/bing/PycharmProjects/DeepModels/Weather_model', file_dir)
    os.makedirs(path)
    logger = load_logger(path)
    logger.info("Model sets: {}".format(args))

    setup_seed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data prepare
    train_set = WeatherDataset(path=args.train_path, model=args.mode, state='train')
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=16,
                              pin_memory=True, drop_last=True)
    valid_set = WeatherDataset(path=args.valid_path, model=args.mode, state='test')
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=True, num_workers=16,
                              pin_memory=True)

    model = Model(
        in_chan=1, encoder_chan_rise=32, time_steps=20, generate_channels=[8, 16, 32, 128, 512],
        num_l_blocks=4, input_shape=(8, 15, 18), h_state_dims=[256, 128, 64, 32], x_dims=[512, 256, 128, 64]
    )

    logger.info("Model architecture: \n {}".format(model))
    logger.info("Model have {}M parameters in total".format(sum(x.numel() for x in model.parameters()) / 1000000))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()
    scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.num_epochs-args.warm_up_epochs),
                                                               eta_min=1e-6, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=1, total_epoch=args.warm_up_epochs,
                                       after_scheduler=scheduler_cos)
    scheduler.step()

    train(num_epochs=args.num_epochs, train_loader=train_loader, valid_loader=valid_loader, model=model,
          criterion=criterion, optimizer=optimizer, devices=devices, scheduler=scheduler, resume=args.resume,
          model_path=args.checkpoint_path, parallel=args.parallel, path=path, logger=logger)
