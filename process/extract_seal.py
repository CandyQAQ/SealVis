import os
import cv2
import numpy as np


def extract(img):
    data = np.float32(img.reshape((-1, 3)))
    h, w, ch = img.shape

    for y in range(w):
        for x in range(h):
            if img[x][y][0] <= 135 and img[x][y][1] <= 130 and img[x][y][2] <= 130:
                img[x, y] = (255, 255, 255)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(data, 2, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)

    index = label[0][0]
    mask = np.ones((h, w), dtype=np.uint8) * 255.
    label = np.reshape(label, (h, w))
    mask[label == index] = 0

    alpha = mask.astype(np.float32) / 255.
    fg = alpha[..., None] * img
    bg = np.ones(img.shape, dtype=np.float) * 255.

    result = fg + (1 - alpha[..., None]) * bg
    # result = cv2.resize(result, (224, 224))

    return result


def red_image(img):
    data = np.float32(img.reshape((-1, 3)))
    h, w, ch = img.shape

    # for x in range(h):
    #     for y in range(w):
    #         if img[x][y][0] < 50 and img[x][y][1] < 50 and img[x][y][2] < 50:
    #             img[x][y][0], img[x][y][1], img[x][y][2] = 255, 255, 255
    #         img[x][y][2] = int(img[x][y][2] * 1.05)
    #         if img[x][y][2] > 255:
    #             img[x][y][2] = 255

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    num_clusters = 2
    ret, label, center = cv2.kmeans(data, num_clusters, None, criteria, num_clusters, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)

    label_reshape = label.reshape(h, w)
    axis_0 = np.mean(img[np.where(label_reshape == 0)])
    axis_1 = np.mean(img[np.where(label_reshape == 1)])
    # print(axis_0, axis_1)
    if axis_0 > axis_1:
        color = np.uint8([[255, 255, 255],
                          [0, 0, 255]])
    else:
        color = np.uint8([[0, 0, 255],
                          [255, 255, 255]])

    res = color[label.flatten()]
    result = res.reshape((img.shape))

    return result


def denoise_image(img):
    h, w, ch = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    for i in range(1, h-1):
        for j in range(1, w-1):
            if binary[i][j] == 0 and np.mean([binary[i-1][j-1], binary[i-1][j], binary[i-1][j+1], binary[i][j-1], binary[i][j+1], binary[i+1][j-1], binary[i+1][j], binary[i+1][j+1]]) / 255 > 0.8:
                binary[i][j] = 255

    return binary


def write_image(root, path, img):
    cv2.imwrite(os.path.join(root, path), img)


origin_root = "../back/data/Seals/"
extract_root = "../back/data/Extracted/"
red_root = "../back/data/Red/"
denoise_root = "../back/data/Denoised/"
bin_root = "../back/data/Bin/"

img_list = os.listdir(origin_root)

for img_path in img_list:
    original_img = cv2.imread(os.path.join(origin_root, img_path))
    # original_img = cv2.imdecode(np.fromfile(os.path.join(origin_root, img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    ex_img = extract(original_img)
    # red_img = red_image(original_img)
    # denoise_img = denoise_image(red_img)
    write_image(extract_root, img_path, ex_img)
    # write_image(red_root, img_path, red_img)
    # write_image(denoise_root, img_path, denoise_img)
