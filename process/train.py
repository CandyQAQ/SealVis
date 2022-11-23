import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, Dataset
from network import VggNet
from data import Data

print(torch.cuda.is_available())

# 1. prepare data
print("Start loading training data")
root = "../back/data/dataset/Train/"
train_dataset = Data(root, False)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
print("Finish loading training data")

print("Start loading value data")
root_val = "../back/data/dataset/Test/"
val_dataset = Data(root_val, False)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
print("Finish loading value data")

# 2. load model
print("Start loading model")
model = VggNet(num_classes=1148)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 3. prepare super parameters
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 4. train
print("Start training")
val_acc_list = []
for epoch in range(100):
    model.train()
    train_loss = 0.0
    start_epoch = time.time()
    print(epoch, "epoch", "length:", len(train_dataloader))
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # print("current batch:", batch_idx, "current loss:", train_loss)

    # val
    print("Start evaluating")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc_val = correct / total
    val_acc_list.append(acc_val)

    # save model
    torch.save(model.state_dict(), './checkpoint/model-epoch-%s.pth' % epoch)
    if acc_val == max(val_acc_list):
        torch.save(model.state_dict(), "./checkpoint/best.pth")
        print("save epoch {} model".format(epoch))

    print("epoch = {},  loss = {},  acc_val = {}".format(epoch, train_loss, acc_val))
