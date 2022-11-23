from flask import Flask
from flask import jsonify, request
from flask_cors import CORS
import pymongo
import cv2
import json
import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from analyze.network import VggNet


app = Flask(__name__)
# cors = CORS(app, resources={r"/getMsg": {"origins": "*"}})
CORS(app, resources=r'/*')

app.config['DEBUG'] = True
mongo = pymongo.MongoClient(host='localhost', port=27017)
db = mongo.ShiTao
collection = db.Relations
collection_images = db.Images
collection_features = db.Features
collection_experience = db.Experience
collection_seals = db.Seals


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


def gradcam(seal_name, output_dir):
    fmap_block = list()
    grad_block = list()

    img = cv2.imread("./data/Seals/" + seal_name, 1)
    img = cv2.resize(img, (224, 224))
    img_input = img_preprocess(img)

    net = VggNet(num_classes=1148)

    pthfile = './analyze/model-epoch-51.pth'
    net.load_state_dict(torch.load(pthfile, map_location='cpu'), strict=False)

    net.eval()

    def backward_hook(module, grad_in, grad_out):
        grad_block.append(grad_out[0].detach())

    def farward_hook(module, input, output):
        fmap_block.append(output)


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


def sift(img, path):
    img = cv2.resize(src=img, dsize=(224, 224))
    sift = cv2.SIFT_create()
    kp = sift.detect(img, None)
    cv2.drawKeypoints(image=img, keypoints=kp, outImage=img, color=(0, 0, 0))

    cv2.imwrite(path, img)


def corner(img, path):
    img = cv2.resize(src=img, dsize=(224, 224))
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2GRAY)
    tomasiCorners = cv2.goodFeaturesToTrack(image=gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
    tomasiCorners = np.int0(tomasiCorners)
    for corner in tomasiCorners:
        x, y = corner.ravel()
        cv2.circle(img=img, center=(x, y), radius=3, color=(34, 76, 52), thickness=-1)

    cv2.imwrite(path, img)


def outline(img, path):
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

    hull = cv2.convexHull(np.array(points))
    cv2.polylines(img, [hull], True, (0, 0, 0), 3)

    cv2.imwrite(path, img)


def make_graphs(seal_name, interpretation):
    print(seal_name, interpretation)
    with open("D://Research//Project//front//public//graph_all.json", 'r', encoding='utf-8') as json_file:
        graph_json = json.load(json_file)
    graph_nodes, graph_edges, id_list = [], [], []
    for node in graph_json["nodes"]:
        if node["label"] == interpretation:
            if node["img"] == seal_name:
                seal_id = node["id"]
                node["x"] = 750
                node["y"] = 50
            graph_nodes.append(node)
            if node["id"] not in id_list:
                id_list.append(node["id"])
    for edge in graph_json["edges"]:
        if edge["source"] in id_list and edge["target"] in id_list and edge["mode"] == "0":
            graph_edges.append(edge)
        if edge["source"] == seal_id or edge["target"] == seal_id:
            graph_edges.append(edge)
    new_graph = {"nodes": graph_nodes, "edges": graph_edges}
    with open('D://Research//Project//front//public//graph.json', 'w', encoding='utf-8') as f:
        json.dump(new_graph, f)

    with open("D://Research/Project//front//public//graph_all_similarity.json", 'r', encoding='utf-8') as json_file:
        graph_json = json.load(json_file)
    graph_nodes, graph_edges, id_list = [], [], []
    for node in graph_json["nodes"]:
        if node["label"] == interpretation:
            if node["img"] == seal_name:
                seal_id = node["id"]
                node["x"] = 750
                node["y"] = 50
            graph_nodes.append(node)
            if node["id"] not in id_list:
                id_list.append(node["id"])
    max_edge = 0
    for edge in graph_json["edges"]:
        if edge["source"] == seal_id and edge["similarity"] != "-1":
            edge["weight"] = int(edge["similarity"])
            if max_edge < edge["weight"]:
                max_edge = edge["weight"]
    for edge in graph_json["edges"]:
        if edge["source"] == seal_id and edge["similarity"] != "-1":
            edge["weight"] = max_edge - edge["weight"] + 1
            graph_edges.append(edge)
    new_graph = {"nodes": graph_nodes, "edges": graph_edges}
    with open('D://Research/Project//front//public//graph_similarity.json', 'w', encoding='utf-8') as f:
        json.dump(new_graph, f)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/getImageInfo', methods=['GET', 'POST'])
def images_info():
    # users = mongo.ShiTao.Seals.find_one({'image_name': '000001.jpg'})
    img_name = request.args.get('image_name')
    users = collection_images.find({'image_name': img_name})
    data = []
    for user in users:
        data.append({'name': user['name'], 'description': user['description']})
    response = {
        'msg': 'success',
        'data': data
    }
    return jsonify(response)


def get_first_seal(seal_name):
    users = collection_features.find({'seal_name': seal_name})
    data = []
    for user in users:
        print(user['similar_cls'])
        for cls in user['similar_cls']:
            seals = collection.find({'seal_name': cls})
            for seal in seals:
                if seal['seal_name'] != seal_name:
                    original_images = collection_images.find({'image_name': seal['image_name']})
                    for image in original_images:
                        data.append({'seal_name': seal['seal_name'], 'seal_interpretation': seal['seal_interpretation'],
                                     'processed_name': seal['processed_name'], 'image_name': seal['image_name'],
                                     'name': image['name']})
    if len(data) == 0:
        return "collection"

    return data[0]['seal_name']


@app.route('/getFeatures', methods=['GET', 'POST'])
def get_features():
    seal1_name = request.args.get('seal_name')
    seal2_name = get_first_seal(seal1_name)

    if seal2_name == "collection":
        response = {
            'msg': 'collection',
            'data': []
        }
        return jsonify(response)

    users = collection_features.find({'seal_name': seal1_name})
    for user in users:
        data = user['similar_cls'][seal2_name]
        break
    print(data)

    response = {
        'msg': 'success',
        'data': data
    }

    return jsonify(response)


@app.route('/getSimilarity', methods=['GET', 'POST'])
def get_similarity():
    seal_name = request.args.get('seal_name')

    data = ""

    users = collection_features.find({'seal_name': seal_name})
    for user in users:
        data = user['confidence']
        print("confidence:", data)
        break
    print(data)

    response = {
        'msg': 'success',
        'data': data
    }
    return jsonify(response)


@app.route('/getOtherFeatures', methods=['GET', 'POST'])
def get_other_features():
    seal1_name = request.args.get('seal1_name')
    seal2_name = request.args.get('seal2_name')

    users = collection_features.find({'seal_name': seal1_name})
    for user in users:
        data = user['similar_cls'][seal2_name]
        break
    print(data)

    response = {
        'msg': 'success',
        'data': data
    }
    return jsonify(response)


@app.route('/analyzeInfo', methods=['GET', 'POST'])
def analyze_info():
    interpretation = request.args.get('interpretation')
    seal1_name = request.args.get('seal1_name')
    seal1_origin = cv2.imread("./data/Seals/" + seal1_name)
    seal1_extracted = cv2.imread("./data/Extracted/" + seal1_name)

    seal2_name = get_first_seal(seal1_name)
    seal2_origin = cv2.imread("./data/Seals/" + seal2_name)
    seal2_extracted = cv2.imread("./data/Extracted/" + seal2_name)

    if seal2_name == "collection":
        response = {
            'msg': 'collection'
        }
        return jsonify(response)

    gradcam(seal1_name, "../front/src/assets/gradcam_1.jpg")
    gradcam(seal2_name, "../front/src/assets/gradcam_2.jpg")

    sift(seal1_origin, "../front/src/assets/sift_1.jpg")
    sift(seal2_origin, "../front/src/assets/sift_2.jpg")

    corner(seal1_extracted, "../front/src/assets/corner_1.jpg")
    corner(seal2_extracted, "../front/src/assets/corner_2.jpg")

    outline(seal1_extracted, "../front/src/assets/outline_1.jpg")
    outline(seal2_extracted, "../front/src/assets/outline_2.jpg")

    response = {
        'msg': 'success'
    }

    return jsonify(response)


@app.route('/analyzeSeal', methods=['GET', 'POST'])
def analyze_seal():
    # interpretation = request.args.get('interpretation')
    seal1_name = request.args.get('seal1_name')
    seal1_origin = cv2.imread("./data/Seals/" + seal1_name)
    seal1_extracted = cv2.imread("./data/Extracted/" + seal1_name)
    seal2_name = request.args.get('seal2_name')
    seal2_origin = cv2.imread("./data/Seals/" + seal2_name)
    seal2_extracted = cv2.imread("./data/Extracted/" + seal2_name)

    gradcam(seal1_name, "../front/src/assets/gradcam_1.jpg")
    gradcam(seal2_name, "../front/src/assets/gradcam_2.jpg")

    sift(seal1_origin, "../front/src/assets/sift_1.jpg")
    sift(seal2_origin, "../front/src/assets/sift_2.jpg")

    corner(seal1_extracted, "../front/src/assets/corner_1.jpg")
    corner(seal2_extracted, "../front/src/assets/corner_2.jpg")

    outline(seal1_extracted, "../front/src/assets/outline_1.jpg")
    outline(seal2_extracted, "../front/src/assets/outline_2.jpg")

    # make_graphs(seal1_name, interpretation)

    response = {
        'msg': 'success'
    }
    return jsonify(response)


@app.route('/changeMark', methods=['GET', 'POST'])
def change_mark():
    seal_name = request.args.get('seal_name')
    mark = request.args.get('mark')

    collection.update_one(
        {'seal_name': seal_name},
        {'$set': {"mark": mark}},
        False,
        True
    )

    data = []
    response = {
        'msg': 'success',
        'data': data
    }
    return jsonify(response)


@app.route('/getSeals', methods=['GET', 'POST'])
def images():
    # users = mongo.ShiTao.Seals.find_one({'image_name': '000001.jpg'})
    img_name = request.args.get('img_name')
    users = collection.find({'image_name': img_name})
    data = []
    for user in users:
        data.append({'seal_name': user['seal_name'], 'seal_interpretation': user['seal_interpretation'],
                     'processed_name': user['processed_name'], 'mark': user['mark']})
    response = {
        'msg': 'success',
        'data': data
    }
    return jsonify(response)


@app.route('/getSimilarSeals', methods=['GET', 'POST'])
def similar_seals():
    seal_name = request.args.get('seal_name')
    users = collection_features.find({'seal_name': seal_name})
    data = []
    for user in users:
        for cls in user['similar_cls']:
            seals = collection.find({'seal_name': cls})
            for seal in seals:
                if seal['seal_name'] != seal_name:
                    original_images = collection_images.find({'image_name': seal['image_name']})
                    for image in original_images:
                        if image['standard'] == "TRUE":
                            data.append({'seal_name': seal['seal_name'], 'seal_interpretation': seal['seal_interpretation'],
                                         'processed_name': seal['processed_name'], 'image_name': seal['image_name'], 'name': image['name'], 'features': user['similar_cls'][cls]})
    response = {
        'msg': 'success',
        'data': data
    }
    return jsonify(response)


@app.route('/getTimeData', methods=['GET', 'POST'])
def time_data():
    data = []
    for years in collection_experience.find():
        infos = collection_seals.find({'year': years['year']})
        seal_list = []
        interpretation = []
        for info in infos:
            seals = collection.find({'seal_name': info['seal_name']})
            for seal in seals:
                if info['seal_interpretation'] not in interpretation and seal['property'] == 'personal':
                    seal_list.append({'seal_name': info['seal_name'], 'seal_interpretation': info['seal_interpretation'], 'select': 'false'})
                    interpretation.append(info['seal_interpretation'])
            if len(seal_list) >= 7:
                break
            seal_list = sorted(seal_list, key=lambda x: x['seal_interpretation'])
        data.append({'year': years['year'], 'experience': years['experience'], 'seal_list': seal_list})

    response = {
        'msg': 'success',
        'data': data
    }
    return jsonify(response)


@app.route('/getSealData', methods=['GET', 'POST'])
def seal_data():
    data = []
    interpretations = ['原濟', '石濤', '老濤', '清湘石濤', '濟山僧', '前有龍眠', '弄墨人老濤', '前有龍眠濟', '苦瓜和尚濟畫法', '苦瓜', '清湘濟', '苦瓜和尚',
                       '善果月之子天童忞之孫原濟之章', '清湘老人', '元濟', '得一人知己無憾', '清湘石道人', '苦瓜龢尚', '癡絶', '臣僧元濟', '瞎尊者', '頭白依然不識字',
                       '釋元濟印', '膏肓子濟', '眼中之人吾老矣', '若極', '零丁老人']
    for interpretation in interpretations:
        for seals in collection_seals.find():
            if seals['seal_interpretation'] == interpretation:
                data.append({'seal_name': seals['seal_name'], 'seal_interpretation': interpretation})
                break
    print(data)

    response = {
        'msg': 'success',
        'data': data
    }
    return jsonify(response)


@app.route('/calculateSimilarity', methods=['GET', 'POST'])
def calculate_similarity():
    seal1_name = request.args.get('seal1_name')
    seal2_name = request.args.get('seal2_name')

    data = []
    features = []

    users = collection_features.find({'seal_name': seal1_name})
    try:
        for user in users:
            features = user['similar_cls'][seal2_name]
            break
        data.append(features[0]['value'] * 0.32 + features[1]['value'] * 0.1 + features[2]['value'] * 0.23 + features[3]['value'] * 0.05 + features[4]['value'] * 0.3)
    except:
        data.append(0)

    response = {
        'msg': 'success',
        'data': data
    }
    return jsonify(response)


@app.route('/getOriginalImage', methods=['GET', 'POST'])
def get_original_images():
    seal1_name = request.args.get('seal1_name')
    seal2_name = request.args.get('seal2_name')
    users = collection.find({'seal_name': seal1_name})
    data = []
    for user in users:
        data.append({'image1_name': user['image_name']})
    users = collection.find({'seal_name': seal2_name})
    for user in users:
        data.append({'image2_name': user['image_name']})
    response = {
        'msg': 'success',
        'data': data
    }
    return jsonify(response)


@app.route('/makeGraph', methods=['GET', 'POST'])
def make_graph():
    seal_name = request.args.get('seal_name')
    interpretation = request.args.get('interpretation')
    print(seal_name, interpretation)
    with open("D://Research/Project/front/public/graph_all.json", 'r', encoding='utf-8') as json_file:
        graph_json = json.load(json_file)
    graph_nodes, graph_edges, id_list = [], [], []
    for node in graph_json["nodes"]:
        if node["label"] == interpretation:
            if node["img"] == seal_name:
                seal_id = node["id"]
                node["x"] = 750
                node["y"] = 50
            graph_nodes.append(node)
            if node["id"] not in id_list:
                id_list.append(node["id"])
    print("inter", interpretation)
    for edge in graph_json["edges"]:
        if edge["source"] in id_list and edge["target"] in id_list and edge["mode"] == "0":
            graph_edges.append(edge)
        if edge["source"] == seal_id or edge["target"] == seal_id:
            graph_edges.append(edge)
    new_graph = {"nodes": graph_nodes, "edges": graph_edges}
    with open('D://Research//Project/front/public/graph.json', 'w', encoding='utf-8') as f:
        json.dump(new_graph, f)

    with open("D://Research//Project/front/public/graph_all_similarity.json", 'r', encoding='utf-8') as json_file:
        graph_json = json.load(json_file)
    graph_nodes, graph_edges, id_list = [], [], []
    for node in graph_json["nodes"]:
        if node["label"] == interpretation:
            if node["img"] == seal_name:
                seal_id = node["id"]
                node["x"] = 750
                node["y"] = 50
            graph_nodes.append(node)
            if node["id"] not in id_list:
                id_list.append(node["id"])
    max_edge = 0
    for edge in graph_json["edges"]:
        if edge["source"] == seal_id and edge["similarity"] != "-1":
            edge["weight"] = int(edge["similarity"])
            if max_edge < edge["weight"]:
                max_edge = edge["weight"]
    for edge in graph_json["edges"]:
        if edge["source"] == seal_id and edge["similarity"] != "-1":
            edge["weight"] = max_edge - edge["weight"] + 1
            graph_edges.append(edge)
    new_graph = {"nodes": graph_nodes, "edges": graph_edges}
    with open('D://Research//Project/front/public/graph_similarity.json', 'w', encoding='utf-8') as f:
        json.dump(new_graph, f)
    response = {
        'msg': 'success'
    }
    return jsonify(response)


@app.route('/getOtherImages', methods=['GET', 'POST'])
def get_other_images():
    image_name = request.args.get('front_page')
    users = collection_images.find({'image_name': image_name})
    data = []
    for user in users:
        images_collection = collection_images.find({'front_page': user['front_page']})
        for image in images_collection:
            data.append({'image_name': image['image_name']})
    response = {
        'msg': 'success',
        'data': data
    }
    print("data:", data)
    return jsonify(response)


@app.route('/getMsg', methods=['GET', 'POST'])
def home():
    # users = mongo.ShiTao.Seals.find_one({'image_name': '000001.jpg'})
    users = collection.find({'seal_name': '000001_1.jpg'})
    print(users)
    for user in users:
        print(user)
    response = {
        'msg': user['seal_path']
        # 'msg': 'Hello, Python !'
    }
    return jsonify(response)


@app.route('/CropSeals', methods=['GET', 'POST'])
def crop_seals():
    seal1_name = request.args.get('seal1_name')
    seal1_img = cv2.imread("./data/Denoised/" + seal1_name)
    seal1_origin = cv2.imread("./data/Seals/" + seal1_name)
    seal2_name = request.args.get('seal2_name')
    seal2_img = cv2.imread("./data/Denoised/" + seal2_name)
    seal2_origin = cv2.imread("./data/Seals/" + seal2_name)
    print(seal1_name, seal2_name)
    seal1_left = float(request.args.get('seal1_left'))
    seal1_right = float(request.args.get('seal1_right'))
    seal1_top = float(request.args.get('seal1_top'))
    seal1_bottom = float(request.args.get('seal1_bottom'))
    seal2_left = float(request.args.get('seal2_left'))
    seal2_right = float(request.args.get('seal2_right'))
    seal2_top = float(request.args.get('seal2_top'))
    seal2_bottom = float(request.args.get('seal2_bottom'))
    crop_left = float(request.args.get('crop_left'))
    crop_right = float(request.args.get('crop_right'))
    crop_top = float(request.args.get('crop_top'))
    crop_bottom = float(request.args.get('crop_bottom'))

    seal1_crop_left = int((crop_left - seal1_left) / (seal1_right - seal1_left) * seal1_img.shape[1])
    seal1_crop_right = int((crop_right - seal1_left) / (seal1_right - seal1_left) * seal1_img.shape[1])
    seal1_crop_top = int((crop_top - seal1_top) / (seal1_bottom - seal1_top) * seal1_img.shape[0])
    seal1_crop_bottom = int((crop_bottom - seal1_top) / (seal1_bottom - seal1_top) * seal1_img.shape[0])

    seal2_crop_left = int((crop_left - seal2_left) / (seal2_right - seal2_left) * seal2_img.shape[1])
    seal2_crop_right = int((crop_right - seal2_left) / (seal2_right - seal2_left) * seal2_img.shape[1])
    seal2_crop_top = int((crop_top - seal2_top) / (seal2_bottom - seal2_top) * seal2_img.shape[0])
    seal2_crop_bottom = int((crop_bottom - seal2_top) / (seal2_bottom - seal2_top) * seal2_img.shape[0])

    seal1_crop = seal1_img[seal1_crop_top:seal1_crop_bottom, seal1_crop_left:seal1_crop_right]
    seal2_crop = seal2_img[seal2_crop_top:seal2_crop_bottom, seal2_crop_left:seal2_crop_right]
    seal1_crop_origin = seal1_origin[seal1_crop_top:seal1_crop_bottom, seal1_crop_left:seal1_crop_right]
    seal2_crop_origin = seal2_origin[seal2_crop_top:seal2_crop_bottom, seal2_crop_left:seal2_crop_right]
    cv2.imwrite("../front/src/assets/seal1_crop.jpg", seal1_crop_origin)
    seal2_crop = cv2.resize(seal2_crop, (seal1_crop.shape[1], seal1_crop.shape[0]))
    cv2.imwrite("../front/src/assets/seal2_crop.jpg", seal2_crop_origin)

    result = seal1_crop.copy()
    count, total_1, total_2 = 0, 0, 0
    for x in range(seal1_crop.shape[0]):
        for y in range(seal1_crop.shape[1]):
            if seal1_crop[x][y][0] <= 175 and seal1_crop[x][y][1] <= 175:
                total_1 += 1
            if seal2_crop[x][y][0] <= 175 and seal2_crop[x][y][1] <= 175:
                total_2 += 1
            if seal1_crop[x][y][0] <= 175 and seal1_crop[x][y][1] <= 175 and seal2_crop[x][y][0] <= 175 and seal2_crop[x][y][1] <= 175:
                result[x][y] = (0, 0, 255)
                count += 1
            else:
                result[x][y] = (255, 255, 255)

    print("The similarity is :", int(count / min(total_1, total_2) * 100), "%")
    cv2.imwrite("../front/src/assets/result.jpg", result)

    data = [{'accuracy': int(count / min(total_1, total_2) * 100)}]
    response = {
        'msg': 'success',
        'data': data
    }
    print("data:", int(count / min(total_1, total_2) * 100))
    return jsonify(response)


if __name__ == '__main__':
    app.run()
