import os
from flask import Flask
from flask_cors import CORS
import pymongo
import shutil
import numpy as np
import cv2
import csv


def original_seal_class():
    seal_list = []
    class_index = -1
    line = []
    row = ['seal_name', 'seal_cls', 'index_name', 'seal_interpretation']
    line.append(row)
    for processed_seal in processed_list:
        classes = collection.find({'processed_name': processed_seal})
        if classes.count() > 0:
            class_index += 1
            os.mkdir(os.path.join(dataset_path, str(class_index)))
        seal_index = 0
        for class_seal in classes:
            row = []
            seal_name = str(seal_index) + '.jpg'
            src = os.path.join(cropped_path, class_seal['seal_name'])
            dst = os.path.join(os.path.join(dataset_path, str(class_index)), seal_name)
            shutil.copy(src, dst)
            shutil.copy(src, os.path.join(value_path, class_seal['seal_name']))
            shutil.copy(src, os.path.join(test_path, str(class_index) + '_' + seal_name))
            seal_index += 1
            row.append(class_seal['seal_name'])
            row.append(class_index)
            row.append(seal_name)
            row.append(class_seal['seal_interpretation'])
            seal_list.append(class_seal['seal_name'])
            print(row)
            line.append(row)
            original_seal = cv2.imread(dst)
            extract_seal = extract(original_seal)
            cv2.imwrite(os.path.join(os.path.join(dataset_path, str(class_index)), str(seal_index) + '.jpg'),
                        extract_seal)
            seal_index += 1
            for background in background_list:
                bg = cv2.imread(os.path.join(background_path, background))
                temp_seal = extract_seal.copy()
                add_bg_seal = add_bg(temp_seal, bg)
                cv2.imwrite(os.path.join(os.path.join(dataset_path, str(class_index)), str(seal_index) + '.jpg'),
                            add_bg_seal)
                seal_index += 1
    print(len(line))
    # for seal in cropped_list:
       #  if seal not in seal_list:
           #  print(seal)
    with open('features.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(line)


def delete_empty_dir():
    dir_list = os.listdir(dataset_path)
    for dir in dir_list:
        if not os.listdir(os.path.join(dataset_path, dir)):
            print(os.path.join(dataset_path, dir))
            os.rmdir(os.path.join(dataset_path, dir))


def extract(img):
    data = np.float32(img.reshape((-1, 3)))
    h, w, ch = img.shape
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

    return result


def add_bg(img, bg):
    h, w = img.shape[0], img.shape[1]
    if h > 200 or w > 200:
        img = cv2.resize(img, (224, 224))
        h, w = img.shape[0], img.shape[1]

    for i in range(h):
        for j in range(w):
            if img[i][j][0] == 255 and img[i][j][1] == 255 and img[i][j][2] == 255:
                img[i][j] = bg[i][j]

    return img


def resize_seal():
    seal_classes_list = os.listdir(dataset_path)
    for seal_classes in seal_classes_list:
        seal_list = os.listdir(os.path.join(dataset_path, seal_classes))
        for seal_img in seal_list:
            seal = cv2.imread(os.path.join(os.path.join(dataset_path, seal_classes), seal_img))
            seal = cv2.resize(seal, (224, 224))
            cv2.imwrite(os.path.join(os.path.join(dataset_path, seal_classes), seal_img), seal)


def resize_test_seal():
    seal_list = os.listdir(test_path)
    for seal_img in seal_list:
        seal = cv2.imread(os.path.join(test_path, seal_img))
        seal = cv2.resize(seal, (224, 224))
        cv2.imwrite(os.path.join(test_path, seal_img), seal)


def rename_seal():
    seal_classes_list = os.listdir(dataset_path)
    for seal_classes in seal_classes_list:
        seal_list = os.listdir(os.path.join(dataset_path, seal_classes))
        seal_index = 0
        for seal_img in seal_list:
            seal_name = seal_classes + '_' + str(seal_index).zfill(4) + '.jpg'
            src = os.path.join(os.path.join(dataset_path, seal_classes), seal_img)
            dst = os.path.join(train_path, seal_name)
            shutil.copy(src, dst)
            seal_index += 1


def create_dataset():
    images = os.listdir(train_path)

    i = 0
    for im in images:
        if (i + 1) % 9 == 0:
            shutil.move(train_path + im, value_path + im)
        i += 1


app = Flask(__name__)
CORS(app, resources=r'/*')

app.config['DEBUG'] = True
mongo = pymongo.MongoClient(host='localhost', port=27017)
db = mongo.ShiTao
collection = db.Relations
collection_images = db.Images

cropped_path = "../back/data/Seals/"
processed_path = "../back/data/Processed/"
dataset_path = "../back/data/dataset/original_seals/"
background_path = "../back/data/dataset/background/"
train_path = "../back/data/dataset/Train/"
value_path = "../back/data/dataset/Value/"
test_path = "../back/data/dataset/Test/"
cropped_list = os.listdir(cropped_path)
processed_list = os.listdir(processed_path)
background_list = os.listdir(background_path)

original_seal_class()
delete_empty_dir()
resize_seal()
resize_test_seal()
rename_seal()
create_dataset()
