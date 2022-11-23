import torch
import argparse
from PIL import Image
from network import VggNet
from data import Data
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default='',
                    help='Give a model to test')
args = parser.parse_args()

# load test images
test_root = "../back/data/dataset/Test/"
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
predict_list = []
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_dataloader):
        data, target = data.to(device), list(target)
        output = model(data)
        sorted_list, indices = torch.sort(output.data, dim=1, descending=True)
        for i in range(len(target)):
            cls = target[i].split('_')[0]
            if cls != current_cls:
                result[current_cls] = predict_list
                predict_list = []
                current_cls = cls
            predict_list.append(indices[i].tolist())
    result[current_cls] = predict_list
    with open("predict.txt", "w") as file:
        for key, value in result.items():
            file.writelines([str(key), '\t', str(value), '\n'])
    file.close()
