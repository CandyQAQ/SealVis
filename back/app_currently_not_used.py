from flask import Flask
from flask import jsonify, request
from flask_cors import CORS
import pymongo
import cv2
import json

app = Flask(__name__)
# cors = CORS(app, resources={r"/getMsg": {"origins": "*"}})
CORS(app, resources=r'/*')

app.config['DEBUG'] = True
mongo = pymongo.MongoClient(host='localhost', port=27017)
db = mongo.ShiTao
collection = db.Relations
collection_images = db.Images
collection_features = db.Features


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
    print(data)
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
                     'processed_name': user['processed_name']})
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
                        data.append({'seal_name': seal['seal_name'], 'seal_interpretation': seal['seal_interpretation'],
                                     'processed_name': seal['processed_name'], 'image_name': seal['image_name'], 'name': image['name']})
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
    with open("D:/Research/Project/front/public/graph_all.json", 'r', encoding='utf-8') as json_file:
        graph_json = json.load(json_file)
    graph_nodes, graph_edges, id_list = [], [], []
    print(seal_name)
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
    with open('D:/Research/Project/front/public/graph.json', 'w', encoding='utf-8') as f:
        json.dump(new_graph, f)

    with open("D:/Research/Project/front/public/graph_all_similarity.json", 'r', encoding='utf-8') as json_file:
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
    print(graph_json["edges"])
    for edge in graph_json["edges"]:
        if edge["source"] == seal_id and edge["similarity"] != "-1":
            edge["weight"] = max_edge - edge["weight"] + 1
            graph_edges.append(edge)
    new_graph = {"nodes": graph_nodes, "edges": graph_edges}
    with open('D:/Research/Project/front/public/graph_similarity.json', 'w', encoding='utf-8') as f:
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


# 启动运行
if __name__ == '__main__':
    app.run()