import os

import torch
import argparse
from PIL import Image
from network import VggNet
from data import Data
from torch.utils.data import DataLoader
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default='',
                    help='Give a model to test')
args = parser.parse_args()

test_root = "../back/data/Resized/"
# images = os.listdir(test_root)
# for image in images:
#     img = cv2.imread(os.path.join(test_root, image))
#     img = cv2.resize(img, (224, 224))
#     cv2.imwrite(os.path.join(test_root, image), img)

# load test images
test_dataset = Data(test_root, True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# load model
saved_model = torch.load(args.model)
model = VggNet(num_classes=1148)
model.load_state_dict(saved_model)
device = torch.device('cpu')
model = model.to(device)

# test
print("Start testing")
model.eval()
right, total, current_cls = 0, 0, '0'
result, final_result = {}, {}
features_all = []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_dataloader):
        data, target = data.to(device), list(target)
        output = model(data)
        sorted_list, indices = torch.sort(output.data, dim=1, descending=True)
        new_list = torch.narrow(sorted_list, 1, 0, 2)
        for i in range(len(new_list)):
            features_all.append(new_list[i].tolist())
    with open("features_2.txt", "w") as file:
        for i in range(len(features_all)):
            file.writelines([str(i), '\t', str(features_all[i]), '\n'])
    file.close()

