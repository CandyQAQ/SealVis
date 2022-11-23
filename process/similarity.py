import os
from flask import Flask, request
import json
from flask_cors import CORS
import pymongo
import math
import cv2
import numpy
from scipy.spatial.distance import directed_hausdorff


def correspond_cls(collection):
    cls = {}
    for seals in collection.find():
        cls[seals['seal_name']] = seals['seal_cls']

    return cls


def calculate_cos(vector_1, vector_2):
    numerator, den_seal1, den_seal2 = 0, 0, 0
    for i in range(len(vector_1)):
        numerator += vector_1[i] * vector_2[i]
        den_seal1 += vector_1[i] * vector_1[i]
        den_seal2 += vector_2[i] * vector_2[i]

    cos = numerator / (math.sqrt(den_seal1) * math.sqrt(den_seal2)) * 100

    return cos


def read_vgg_features():
    features = {}
    feature_list = []
    with open("features_2.txt", "r") as file:
        line_list = file.readlines()
    for line in line_list:
        key, value = line.split('\t')[0], eval(line.split('\t')[1].split('\n')[0])
        features[key] = value
        feature_list.append(value)

    # min_1 = min([i[0] for i in feature_list])
    # max_1 = max([i[0] for i in feature_list])
    # min_2 = min([i[1] for i in feature_list])
    # max_2 = max([i[1] for i in feature_list])
    #
    # for key, value in features.items():
    #     features[key] = [(value[0] - min_1) / (max_1 - min_1) * 100 + 0.0001, (value[1] - min_2) / (max_2 - min_2) * 100 + 0.0001]
    #
    # print(features)

    min_cos, max_cos = 100, 0

    for i in range(0, len(feature_list)-1):
        for j in range(1, len(feature_list)):
            cos = calculate_cos(feature_list[i], feature_list[j])
            if cos < min_cos:
                min_cos = cos
            if cos > max_cos:
                max_cos = cos

    return features, 97, max_cos


def vgg_features(seal1_name, seal2_name, cls, vgg_features_all, min_cos, max_cos):
    seal1 = vgg_features_all[cls[seal1_name]]
    seal2 = vgg_features_all[cls[seal2_name]]
    numerator, den_seal1, den_seal2 = 0, 0, 0
    for i in range(len(seal1)):
        numerator += seal1[i] * seal2[i]
        den_seal1 += seal1[i] * seal1[i]
        den_seal2 += seal2[i] * seal2[i]
    similarity = numerator / (math.sqrt(den_seal1) * math.sqrt(den_seal2)) * 100
    similarity = (similarity - min_cos) / (max_cos - min_cos) * 100
    if similarity < 0:
        similarity = 0

    return round(similarity, 2)


def getMatchNum(matches, ratio):
    matchesMask = [[0, 0] for i in range(len(matches))]
    matchNum = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio*n.distance:
            matchesMask[i] = [1, 0]
            matchNum += 1
    return (matchNum, matchesMask)


def sift_features(seal1_name, seal2_name):
    seal1_img = cv2.imread("../back/data/Seals/" + seal1_name)
    seal1_img = cv2.resize(src=seal1_img, dsize=(224, 224))
    seal2_img = cv2.imread("../back/data/Seals/" + seal2_name)
    seal2_img = cv2.resize(src=seal2_img, dsize=(224, 224))

    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=100)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    kp1, des1 = sift.detectAndCompute(seal1_img, None)
    kp2, des2 = sift.detectAndCompute(seal2_img, None)
    try:
        matches = flann.knnMatch(des1, des2, k=2)
        (matchNum, matchesMask) = getMatchNum(matches, 0.9)
        if len(matches) == 0:
            matchRatio = 0
        else:
            matchRatio = matchNum * 100 / len(matches) * 1.5
            if matchRatio > 100:
                matchRatio = 100

        drawParams = dict(matchColor=(0, 255, 0),
                          singlePointColor=(255, 0, 0),
                          matchesMask=matchesMask,
                          flags=0)

        # comparisonImage = cv2.drawMatchesKnn(seal1_img, kp1, seal2_img, kp2, matches, None, **drawParams)
        # cv2.imwrite("sift.jpg", comparisonImage)
        similarity = round(matchRatio, 2)
    except:
        print(seal1_name, seal2_name)
        similarity = 0

    return similarity


def harris_features(seal1_name, seal2_name):
    seal1_img = cv2.imread("../back/data/Extracted/" + seal1_name)
    seal2_img = cv2.imread("../back/data/Extracted/" + seal2_name)

    seal1_img = cv2.resize(src=seal1_img, dsize=(224, 224))
    seal1_gray = cv2.cvtColor(src=seal1_img, code=cv2.COLOR_RGB2GRAY)
    seal1_tomasiCorners = cv2.goodFeaturesToTrack(image=seal1_gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
    seal1_tomasiCorners = numpy.int0(seal1_tomasiCorners)

    seal2_img = cv2.resize(src=seal2_img, dsize=(224, 224))
    seal2_gray = cv2.cvtColor(src=seal2_img, code=cv2.COLOR_RGB2GRAY)
    seal2_tomasiCorners = cv2.goodFeaturesToTrack(image=seal2_gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
    seal2_tomasiCorners = numpy.int0(seal2_tomasiCorners)

    min_corner = min(seal1_tomasiCorners.shape[0], seal2_tomasiCorners.shape[0])
    max_corner = max(seal1_tomasiCorners.shape[0], seal2_tomasiCorners.shape[0])

    similarity = round(seal1_tomasiCorners.shape[0] / seal2_tomasiCorners.shape[0] * 100, 2)
    if similarity > 100:
        similarity = 100

    return similarity


def cal_frechet_distance(curve_a: numpy.ndarray, curve_b: numpy.ndarray):
    def euc_dist(pt1, pt2):
        return numpy.sqrt(numpy.square(pt2[0] - pt1[0]) + numpy.square(pt2[1] - pt1[1]))

    def _c(ca, i, j, P, Q):
        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = euc_dist(P[0], Q[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(ca, i - 1, 0, P, Q), euc_dist(P[i], Q[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(ca, 0, j - 1, P, Q), euc_dist(P[0], Q[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(min(_c(ca, i - 1, j, P, Q),
                               _c(ca, i - 1, j - 1, P, Q),
                               _c(ca, i, j - 1, P, Q)),
                           euc_dist(P[i], Q[j]))
        else:
            ca[i, j] = float("inf")
        return ca[i, j]

    def frechet_distance(P, Q):
        ca = numpy.ones((len(P), len(Q)))
        ca = numpy.multiply(ca, -1)
        dis = _c(ca, len(P) - 1, len(Q) - 1, P, Q)
        return dis

    curve_line_a = list(zip(range(len(curve_a)), curve_a))
    curve_line_b = list(zip(range(len(curve_b)), curve_b))
    return frechet_distance(curve_line_a, curve_line_b)


def get_contours(img):
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    # img_contour, contours = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    # return contours[0]

    img = cv2.resize(src=img, dsize=(224, 224))
    w, h, c = img.shape
    result = img.copy()
    points = []

    for i in range(h):
        for j in range(w):
            if img[i][j][0] < 127:
                cv2.circle(result, (i, j), 1, (255, 0, 0), 2)
                break
        for j in range(w - 1, 0, -1):
            if img[i][j][0] < 127:
                cv2.circle(result, (i, j), 1, (255, 0, 0), 2)
                break

    for i in range(w):
        for j in range(h):
            if img[j][i][0] < 127:
                cv2.circle(result, (j, i), 1, (255, 0, 0), 2)
                break
        for j in range(h - 1, 0, -1):
            if img[j][i][0] < 127:
                cv2.circle(result, (j, i), 1, (255, 0, 0), 2)
                break

    for i in range(h):
        for j in range(w):
            if result[i][j][1] == 0:
                points.append([[i, j]])
                break
        for j in range(w - 1, 0, -1):
            if result[i][j][1] == 0:
                points.append([[i, j]])
                break

    for i in range(w):
        for j in range(h):
            if result[j][i][1] == 0:
                points.append([[j, i]])
                break
        for j in range(h - 1, 0, -1):
            if result[j][i][1] == 0:
                points.append([[j, i]])
                break

    hull = cv2.convexHull(numpy.array(points))
    # print(numpy.array(hull).flatten())

    return numpy.array(hull).flatten()


def outline_features(seal1_name, seal2_name):
    seal1_img = cv2.imread("../back/data/Extracted/" + seal1_name)
    seal2_img = cv2.imread("../back/data/Extracted/" + seal2_name)

    # hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    try:
        seal1_outline = get_contours(seal1_img)
        seal2_outline = get_contours(seal2_img)

        # similarity = 100 - directed_hausdorff(seal1_outline, seal2_outline)[0] * 1.5
        similarity = cal_frechet_distance(seal1_outline, seal2_outline)
        if similarity < 0:
            similarity = 0
        if similarity > 100:
            similarity = 100
    except:
        similarity = 0

    # print(seal1_name, seal2_name, round(similarity, 2))
    return round(similarity, 2)


def calculate_similarity(seal1_name, seal2_name, cls, vgg_features_all, min_cos, max_cos):
    print(seal1_name, seal2_name)
    vgg = vgg_features(seal1_name, seal2_name, cls, vgg_features_all, min_cos, max_cos)
    sift = sift_features(seal1_name, seal2_name)
    harris = harris_features(seal1_name, seal2_name)
    outline = outline_features(seal1_name, seal2_name)

    return vgg, sift, harris, outline


app = Flask(__name__)
CORS(app, resources=r'/*')

app.config['DEBUG'] = True
mongo = pymongo.MongoClient(host='localhost', port=27017)
db = mongo.ShiTao
collection_features = db.Features
collection_images = db.Images
collection_relations = db.Relations

cls = correspond_cls(collection_features)
vgg_features_all, min_cos, max_cos = read_vgg_features()

for seals in collection_features.find(no_cursor_timeout=True, batch_size=10):
    features_sim = {}
    sim_cls = {}
    confidence = 0
    # Find similar seals according to seal interpretation
    similar_seals = collection_relations.find({'seal_interpretation': seals['seal_interpretation']}, no_cursor_timeout=True, batch_size=10)
    for similar_seal in similar_seals:
        if similar_seal['seal_name'] != seals['seal_name']:
            seals_infos = collection_relations.find({'seal_name': seals['seal_name']})
            for seals_info in seals_infos:
                if similar_seal['class'] == seals_info['class']:
                    # print(seals['seal_name'], seals_info['class'], similar_seal['seal_name'], similar_seal['class'])
                    # Select standard seals
                    seals_image = collection_relations.find({'seal_name': similar_seal['seal_name']})
                    for seal_image in seals_image:
                        front_pages = collection_images.find({'front_page': seal_image['front_page']})
                        for front_page in front_pages:
                            standard = front_page['standard']
                            break
                        break
                    if standard == "TRUE":
                        vgg_sim, sift_sim, harris_sim, outline_sim = calculate_similarity(seals['seal_name'],
                                                                                          similar_seal['seal_name'],
                                                                                          cls, vgg_features_all, min_cos,
                                                                                          max_cos)

                        similarity = vgg_sim * 0.3 + sift_sim * 0.2 + harris_sim * 0.26 + outline_sim * 0.24
                        sim_cls[similar_seal['seal_name']] = [vgg_sim, sift_sim, harris_sim, outline_sim]
                        features_sim[similar_seal['seal_name']] = [vgg_sim, sift_sim, harris_sim, outline_sim]
                        confidence += vgg_sim * 0.25 + sift_sim * 0.25 + harris_sim * 0.25 + outline_sim * 0.25

                        # outline_sim = calculate_similarity(seals['seal_name'], similar_seal['seal_name'],
                        #                                   cls, vgg_features_all, min_cos,
                        #                                   max_cos)

    similar_seals.close()
    if len(features_sim) == 0:
        confidence = 0
    else:
        confidence /= len(features_sim)
    collection_features.update_one(
        {'seal_name': seals['seal_name']},
        {'$set': {"mean": round(confidence, 2)}},
        False,
        True
    )
    collection_features.update_one(
        {'seal_name': seals['seal_name']},
        {'$set': {"similar_cls": sim_cls}},
        False,
        True
    )
