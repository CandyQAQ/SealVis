from flask import Flask
from flask_cors import CORS
import pymongo


app = Flask(__name__)
CORS(app, resources=r'/*')

app.config['DEBUG'] = True
mongo = pymongo.MongoClient(host='localhost', port=27017)
db = mongo.ShiTao
collection = db.Relations
collection_features = db.Features

cls_interpretation = {}
for i in range(1148):
    classes = collection_features.find({'seal_cls': str(i)})
    for cls in classes:
        cls_interpretation[i] = cls['seal_interpretation']
        break

with open("predict.txt", "r") as file:
    line_list = file.readlines()

result, final_result = {}, {}

for line in line_list:
    key, value = line.split('\t')[0], eval(line.split('\t')[1].split('\n')[0])
    result[key] = value

for key, value in result.items():
    sort = {}
    for value_list in value:
        for cls in value_list:
            if cls in sort:
                sort[cls] += (1148 - value_list.index(cls))
            else:
                sort[cls] = (1148 - value_list.index(cls))
    sort_result = list(dict(sorted(sort.items(), key=lambda x: x[1], reverse=True)).keys())
    cls_seals = []
    for cls in range(0, len(sort_result)):
        if cls_interpretation[sort_result[cls]] == cls_interpretation[sort_result[0]]:
            seals = collection_features.find({'seal_cls': str(sort_result[cls])})
            for seal in seals:
                cls_seals.append(seal['seal_name'])
    final_result[key] = cls_seals

for key, value in final_result.items():
    classes = collection_features.find({'seal_cls': key})
    collection_features.update_many(
        {'seal_cls': key},
        {'$set': {"similar_cls": value}},
        False,
        True
    )
