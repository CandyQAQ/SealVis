# -*- encoding: utf-8 -*-
import argparse
import torch
import numpy as np
from net import SiameseNetwork
from contrastive import ContrastiveLoss
from PIL import Image
import os
import os.path
import csv
import codecs


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model', '-m', default='',
                        help='Give a model to test')
    parser.add_argument('--train-plot', action='store_true', default=False,
                        help='Plot train loss')
    parser.add_argument('--cls', '-c', type=int, default=1507,
                        help='Number of classes')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    model = SiameseNetwork()
    if args.cuda:
        model.cuda()

    learning_rate = 0.0001
    momentum = 0.9
    criterion = ContrastiveLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    Test_data = Test_Dataset("../back/data/dataset/Test/")
    test_loader = torch.utils.data.DataLoader(
        Test_data, batch_size=args.batchsize, shuffle=True, **kwargs)
    Train_data = Test_Dataset("../back/data/dataset/Train/")
    train_loader = torch.utils.data.DataLoader(
        Train_data, batch_size=args.batchsize, shuffle=True, **kwargs)

    # print(len(test_loader))

    def test(model):
        model.eval()
        train_all = []
        train_labels = []

        for batch_idx, (x, labels) in enumerate(train_loader):
            if args.cuda:
                x, labels = x.cuda(), labels.cuda()
            # x, labels = Variable(x, volatile=True), Variable(labels)
            output = model.forward_once(x)
            train_all.extend(output.data.cpu().numpy().tolist())
            train_labels.extend(labels.data.cpu().numpy().tolist())

        numpy_train = np.array(train_all)
        numpy_train_labels = np.array(train_labels)
        return numpy_train, numpy_train_labels

    data = {}
    with codecs.open('features.csv', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            data[row['seal_cls']] = row['seal_name']

    numpy_train, numpy_train_labels = test(model)
    line = []
    row = ['seal_cls', 'feature']
    line.append(row)
    for i in range(args.cls):
        if i in numpy_train_labels:
            train_f = numpy_train[np.where(numpy_train_labels == i)]
            row = [data[str(i)],
                   [np.mean(train_f[:, 0]), np.mean(train_f[:, 1]), np.mean(train_f[:, 2]), np.mean(train_f[:, 3]),
                    np.mean(train_f[:, 4]), np.mean(train_f[:, 5]), np.mean(train_f[:, 6]), np.mean(train_f[:, 7]),
                    np.mean(train_f[:, 8]), np.mean(train_f[:, 9]), np.mean(train_f[:, 10]), np.mean(train_f[:, 11]),
                    np.mean(train_f[:, 12]), np.mean(train_f[:, 13]), np.mean(train_f[:, 14]),
                    np.mean(train_f[:, 15])]]
            line.append(row)

    with open('test.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(line)


if __name__ == '__main__':
    main()
