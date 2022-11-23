# -*- coding: utf-8 -*-
import json
import time

from py2neo import Graph, Node, Relationship, Subgraph
from py2neo import RelationshipMatcher

# 连接Neo4j
url = "http://localhost:7474"
username = "neo4j"
password = "123456"
graph = Graph(url, auth=(username, password))
print("neo4j info: {}".format(str(graph)))

# 读取数据
with open("data.json", "r", encoding="utf-8") as f:
    data_dict = json.loads(f.read())
nodes = data_dict["nodes"]
relations = data_dict["relations"]

# 查询city和province节点是否在图谱中
cql = "match (n:province) return (n.name);"
province_names = [_["(n.name)"] for _ in graph.run(cql).data()]
cql = "match (n:city) return (n.name);"
city_names = [_["(n.name)"] for _ in graph.run(cql).data()]

# 创建节点
s_time = time.time()
create_node_cnt = 0
create_nodes = []
for node in nodes:
    label = node["label"]
    name = node["name"]
    if label == "city" and name not in city_names:
        attrs = {k: v for k, v in node.items() if k != "label"}
        create_nodes.append(Node(label, **attrs))
        create_node_cnt += 1
    elif label == "province" and name not in province_names:
        attrs = {k: v for k, v in node.items() if k != "label"}
        create_nodes.append(Node(label, **attrs))
        create_node_cnt += 1

# 批量创建节点
batch_size = 50
if create_nodes:
    for i in range(len(create_nodes)//50 + 1):
        subgraph = Subgraph(create_nodes[i*batch_size: (i+1)*batch_size])
        graph.create(subgraph)
        print(f"create {(i+1)*batch_size} nodes")

# 创建关系
cql = "match (n:province) return (n);"
province_nodes = [_["n"] for _ in graph.run(cql).data()]
cql = "match (n:city) return (n);"
city_nodes = [_["n"] for _ in graph.run(cql).data()]
city_dict = {_["name"]: _ for _ in city_nodes}
province_dict = {_["name"]: _ for _ in province_nodes}
create_rel_cnt = 0
create_relations = []
rel_matcher = RelationshipMatcher(graph)
for relation in relations:
    s_node, s_label = relation["subject"], relation["subject_type"]
    e_node, e_label = relation["object"], relation["object_type"]
    rel = relation["predicate"]
    start_node, end_node = None, None
    if s_label == "city":
        start_node = city_dict.get(s_node, None)
    if e_label == "city":
        end_node = city_dict.get(e_node, None)
    elif e_label == "province":
        end_node = province_dict.get(e_node, None)
    if start_node is not None and end_node is not None:
        r_type = rel_matcher.match([start_node, end_node], r_type=rel).first()
        if r_type is None:
            create_relations.append(Relationship(start_node, rel, end_node))
            create_rel_cnt += 1

# 批量创建关系
batch_size = 50
if create_relations:
    for i in range(len(create_relations)//50 + 1):
        subgraph = Subgraph(relationships=create_relations[i*batch_size: (i+1)*batch_size])
        graph.create(subgraph)
        print(f"create {(i+1)*batch_size} relations")

# 输出信息
e_time = time.time()
print(f"create {create_node_cnt} nodes, create {create_rel_cnt} relations.")
print(f"cost time: {round((e_time-s_time)*1000, 4)}ms")
