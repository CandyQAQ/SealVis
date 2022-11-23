import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from network import VggNet


def img_preprocess(img_in):
    img = img_in.copy()
    img = img[:, :, ::-1]
    img = np.ascontiguousarray(img)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    return img


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    fmap_block.append(output)


def cam_show_img(img, feature_map, grads, out_dir, f):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
    grads = grads.reshape([grads.shape[0], -1])
    weights = np.mean(grads, axis=1)
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img

    path_cam_img = os.path.join(out_dir)
    cv2.imwrite(path_cam_img, cam_img)
    print(path_cam_img, "ok")


if __name__ == '__main__':
    # path_img = 'seal.jpg'
    path_img = 'seal_1.jpg'
    output_dir = 'gradcam.jpg'

    classes = [str(i) for i in range(1148)]

    fmap_block = list()
    grad_block = list()

    img = cv2.imread(path_img, 1)
    img = cv2.resize(img, (224, 224))
    img_input = img_preprocess(img)

    net = VggNet(num_classes=1148)

    pthfile = './checkpoint/model-epoch-53.pth'
    net.load_state_dict(torch.load(pthfile), strict=False)

    net.eval()
    print(net)

    net.Conv.register_forward_hook(farward_hook)
    net.Conv.register_backward_hook(backward_hook)

    output = net(img_input)
    idx = np.argmax(output.cpu().data.numpy())

    net.zero_grad()
    class_loss = output[0, idx]
    class_loss.backward()

    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    cam_show_img(img, fmap, grads_val, output_dir, pthfile.split('-')[0])
