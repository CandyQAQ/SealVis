# -*- encoding: utf-8 -*-
import argparse
import torch
import random
import numpy as np
import time
from net import SiameseNetwork
from contrastive import ContrastiveLoss
from torch.autograd import Variable
from PIL import Image
import os
import os.path


class Train_Dataset(object):

    def __init__(self, x0, x1, label):
        self.size = label.shape[0]
        self.x0 = x0
        self.x1 = x1
        self.label = torch.from_numpy(label)

    def __getitem__(self, index):
        image0 = (np.array(Image.open(self.x0[index]).resize((224, 224)), dtype=np.float32) / 255.0).transpose(
            (2, 0, 1))
        image1 = (np.array(Image.open(self.x1[index]).resize((224, 224)), dtype=np.float32) / 255.0).transpose(
            (2, 0, 1))
        return (torch.from_numpy(image0),
                torch.from_numpy(image1),
                self.label[index])

    def __len__(self):
        return self.size


class Test_Dataset(object):
    def __init__(self, file_dir):
        self.file_name = os.listdir(file_dir)
        self.file_list = [os.path.join(file_dir, i) for i in (os.listdir(file_dir))]
        self.size = len(self.file_list)

    def __getitem__(self, index):
        image0 = (np.array(Image.open(self.file_list[index]).resize((224, 224)), dtype=np.float32) / 255.0).transpose(
            (2, 0, 1))
        label = np.array(self.file_name[index].split('_')[0], dtype=np.int32)

        return (torch.from_numpy(image0),
                torch.from_numpy(label))

    def __len__(self):
        return self.size


def create_pairs(data, digit_indices, N):
    x0_data = []
    x1_data = []
    label = []

    n = min([len(digit_indices[d]) for d in range(N)]) - 1
    for d in range(N):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            x0_data.append(data[z1])
            x1_data.append(data[z2])
            label.append(1)

            inc = random.randrange(1, N)
            dn = (d + inc) % N
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            x0_data.append(data[z1])
            x1_data.append(data[z2])
            label.append(0)

    label = np.array(label, dtype=np.int32)
    return x0_data, x1_data, label


def create_iterator(data, label, batchsize, N, shuffle=False):
    digit_indices = [np.where(label == i)[0] for i in range(N)]
    x0, x1, label = create_pairs(data, digit_indices, N)
    ret = Train_Dataset(x0, x1, label)
    return ret


def get_data(file_dir):
    file_list = os.listdir(file_dir)
    data = []
    labels = []
    for file in file_list:
        image_path = os.path.join(file_dir, file)
        label, remain = file.split('_')
        data.append(image_path)
        labels.append(int(label))

    return data, labels


def main():
    N = 1074

    os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_IS"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model', '-m', default='',
                        help='Give a model to test')
    parser.add_argument('--train-plot', action='store_true', default=False,
                        help='Plot train loss')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("Args: %s" % args)

    data, label = get_data(file_dir='../back/data/dataset/Train/')
    train_iter = create_iterator(data, np.array(label), args.batchsize, N)

    # model
    model = SiameseNetwork()
    if args.cuda:
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    learning_rate = 0.0001
    momentum = 0.9
    criterion = ContrastiveLoss()
    # optimizer = torch.optim.Adam(
    #     [p for p in model.parameters() if p.requires_grad],
    #     lr=learning_rate
    # )

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    Test_data = Test_Dataset("../back/data/dataset/Test/")
    test_loader = torch.utils.data.DataLoader(
        Test_data, batch_size=args.batchsize, shuffle=True, **kwargs)
    train_loader = torch.utils.data.DataLoader(
        train_iter,
        batch_size=args.batchsize, shuffle=True, **kwargs)

    def train(epoch):
        train_loss = 0
        model.train()
        start = time.time()
        start_epoch = time.time()
        for batch_idx, (x0, x1, labels) in enumerate(train_loader):
            labels = labels.float()
            if args.cuda:
                x0, x1, labels = x0.cuda(), x1.cuda(), labels.cuda()
            x0, x1, labels = Variable(x0), Variable(x1), Variable(labels)
            output1, output2 = model(x0, x1)
            loss = criterion(output1, output2, labels)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), './checkpoint/model-epoch-%s.pth' % epoch)
        end = time.time()
        took = end - start_epoch
        print('Train epoch: {} \tLoss: {} \tTook:{:.2f}'.format(epoch, train_loss, took))
        return train_loss

    if len(args.model) == 0:
        for epoch in range(1, args.epoch + 1):
            train(epoch)
    else:
        saved_model = torch.load(args.model)
        model = SiameseNetwork()
        model.load_state_dict(saved_model)
        if args.cuda:
            model.cuda()


if __name__ == '__main__':
    main()
