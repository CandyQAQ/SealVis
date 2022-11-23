from flask import Flask
from flask_cors import CORS
import pymongo
import numpy


app = Flask(__name__)
CORS(app, resources=r'/*')

app.config['DEBUG'] = True
mongo = pymongo.MongoClient(host='localhost', port=27017)
db = mongo.ShiTao
collection_features = db.Features
collection_relations = db.Relations

for seals in collection_features.find():
    print(seals['seal_name'])
    relation, count, relation_list = 0, 0, []
    images = collection_relations.find({'seal_name': seals['seal_name']})
    for image in images:
        seals_same_image = collection_relations.find({'front_page': image['front_page']})
        for seal_same_image in seals_same_image:
            if seal_same_image['seal_name'] != seals['seal_name'] and seal_same_image['property'] == "personal":
                seals_info = collection_features.find({'seal_name': seal_same_image['seal_name']})
                for seal_info in seals_info:
                    if seal_info['mean'] > 0:
                        relation += seal_info['mean']
                        relation_list.append(seal_info['mean'])
                        count += 1

    if count <= 0:
        relation = 0
    else:
        relation_list.sort(reverse=False)
        if numpy.mean(relation_list[1:]) - relation_list[0] >= 10:
            relation_list = relation_list[1:]
        relation = sum(relation_list) / len(relation_list)

    result, final_result = {}, {}
    confidence = 0

    for sim_seal in seals['similar_cls']:
        sim_seal_infos = collection_relations.find({'seal_name': sim_seal})
        seals_infos = collection_relations.find({'seal_name': seals['seal_name']})
        for sim_seal_info in sim_seal_infos:
            for seals_info in seals_infos:
                if sim_seal_info['class'] == seals_info['class']:
                    item_1, item_2, item_3, item_4, item_5, item_6 = {}, {}, {}, {}, {}, {}
                    seals['similar_cls'][sim_seal].append(round(relation, 2))
                    item_1['value'] = seals['similar_cls'][sim_seal][0]
                    item_1['name'] = "整体特征"
                    item_2['value'] = seals['similar_cls'][sim_seal][1]
                    item_2['name'] = "局部特征"
                    item_3['value'] = seals['similar_cls'][sim_seal][2]
                    item_3['name'] = "角点特征"
                    item_4['value'] = seals['similar_cls'][sim_seal][3]
                    item_4['name'] = "轮廓特征"
                    item_5['value'] = seals['similar_cls'][sim_seal][4]
                    item_5['name'] = "关联特征"
                    if item_5['value'] == 0:
                        item_6['value'] = round(
                            item_1['value'] * 0.25 + item_2['value'] * 0.25 + item_3['value'] * 0.25 + item_4[
                                'value'] * 0.25, 2)
                    else:
                        item_6['value'] = round(
                            item_1['value'] * 0.2 + item_2['value'] * 0.2 + item_3['value'] * 0.2 + item_4[
                                'value'] * 0.2 + item_5['value'] * 0.2, 2)
                    item_6['name'] = "比对分值"
                    result[sim_seal] = [item_1, item_2, item_3, item_4, item_5, item_6]
                    confidence += item_6['value']
                    # print(seals['similar_cls'][sim_seal])

    if len(result) == 0:
        confidence = 0
    else:
        confidence /= len(result)

    result_sorted = sorted(result.items(), key=lambda x: x[1][5]['value'], reverse=True)
    # print(result_sorted)
    for item in result_sorted:
        final_result[item[0]] = item[1][0:5]
    # print(result)
    # print(final_result)
    # break

    print(seals['seal_name'], round(confidence, 2))

    collection_features.update_one(
        {'seal_name': seals['seal_name']},
        {'$set': {"confidence": round(confidence, 2)}},
        False,
        True
    )
    collection_features.update_one(
        {'seal_name': seals['seal_name']},
        {'$set': {"similar_cls": final_result}},
        False,
        True
    )
