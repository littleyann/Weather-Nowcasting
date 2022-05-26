import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dataset.utils import image_read
from torch import nn
matplotlib.use('Agg')


def train(num_epochs, train_loader, valid_loader, model, criterion, optimizer, devices, scheduler, resume,
          model_path, parallel, path, logger):

    start_epoch = 1

    if parallel:
        model = nn.DataParallel(model).cuda()

    if resume:
        train_state = torch.load(model_path)
        model.load_state_dict(train_state['model_state_dict'])
        optimizer.load_state_dict(train_state['optimizer_state_dict'])
        start_epoch = train_state['epoch'] + 1
        scheduler.load_state_dict(train_state['scheduler'])
        # for epoch in range(1, start_epoch):
        #     scheduler.step()

    for epoch in range(start_epoch, num_epochs+1):
        logger.info("Training epoch: {}".format(epoch))
        model.train()
        for i, (data, ground_truth) in enumerate(train_loader):

            optimizer.zero_grad()
            data = data.unsqueeze(2).to(devices).float()
            ground_truth = ground_truth.unsqueeze(2).to(devices).float()
            predicts = model(data)
            loss = criterion(predicts, ground_truth)

            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                logger.info('Epoch [{}/{}], Step [{}/{}], loss: {:.4f}, lr: {:.10f}'.format(epoch,
                            num_epochs, i + 1, len(train_loader), loss.item(),
                              optimizer.state_dict()['param_groups'][0]['lr']))
            if (i + 1) % 600 == 0:
                plot(model, devices, path, epoch, i)

        scheduler.step()
        if epoch >= 1:
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            path1 = path + '/Model_epoch_' + str(epoch) + '.pth'
            torch.save(checkpoint, path1)

        valid(valid_loader, model, devices, criterion, logger)

    path2 = path + '/Model.pth'
    torch.save(model.module.state_dict(), path2)


def valid(valid_loader, model, devices, criterion, logger):
    model.eval()
    with torch.no_grad():
        loss_item = 0
        num = 0
        for i, (data, ground_truth) in enumerate(valid_loader):
            data = data.unsqueeze(2).to(devices).float()
            ground_truth = ground_truth.unsqueeze(2).to(devices).float()
            predicts = model(data)
            loss = criterion(predicts, ground_truth)

            loss_item = loss_item + loss.item()
            num += 1

        logger.info("The valid loss is: {}}".format(loss_item/num))
    model.train()


def plot(model, devices, path, epoch, iter):
    visual_sample = [
        '31219.png', '31220.png', '31221.png', '31222.png', '31223.png', '31224.png', '31225.png', '31226.png',
        '31227.png', '31228.png', '31229.png', '31230.png', '31231.png', '31232.png', '31233.png', '31234.png',
        '31235.png', '31236.png', '31237.png', '31238.png', '31239.png', '31240.png', '31241.png', '31242.png',
        '31243.png', '31244.png', '31245.png', '31246.png', '31247.png', '31248.png', '31249.png', '31250.png',
        '31251.png', '31252.png', '31253.png', '31254.png', '31255.png', '31256.png', '31257.png', '31258.png'
    ]

    data = np.zeros((40, 480, 560))
    for i, img_name in enumerate(visual_sample):
        img_path = '/home/bing/PycharmProjects/dataSets/weather/TestA/Radar/radar_' + img_name
        data[i, :, :] = image_read(img_path)

    train_data = data[0: 20, :, :]
    test_data = data[20: 40, :, :]
    train_data = torch.from_numpy(train_data)
    train_data = train_data.unsqueeze(0).unsqueeze(2).to(devices).float()

    predicts = model(train_data)
    predicts = predicts.squeeze(2).squeeze(0).cpu().detach().numpy()

    plt.figure(figsize=(14, 14))
    for i in range(20):
        plt.subplot(4, 5, i+1)
        plt.imshow(predicts[i, :, :])

    path1 = path + '/' + str(epoch) + '_predict_' + str(iter) + '.png'
    plt.savefig(path1)

    plt.figure(figsize=(14, 14))
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(test_data[i, :, :])

    path2 = path + '/' + str(epoch) + '_GT_' + str(iter) + '.png'
    plt.savefig(path2)










