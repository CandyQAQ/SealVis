from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import os
from PIL import Image


class Data(Dataset):
    def __init__(self, root, test, transforms=None):
        imgs = []
        for path in os.listdir(root):
            if test == False:
                name = path.split('_')
                label = int(name[0])
                # if label > 5:
                #     print("data label error")
            else:
                label = path

            imgs.append((os.path.join(root, path), label))

        print("Number of data:", len(imgs))
        self.imgs = imgs
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            self.transforms = T.Compose([
                T.Resize(224),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index][0]
        label = self.imgs[index][1]

        data = Image.open(img_path)
        if data.mode != "RGB":
            data = data.convert("RGB")
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    root = "../back/data/dataset/Train/"
    train_dataset = Data(root)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for data, label in train_dataset:
        print(data.shape)
        pass
