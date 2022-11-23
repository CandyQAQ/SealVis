from io import StringIO, BytesIO
from pymongo import MongoClient
import gridfs
import os
import matplotlib.pyplot as plt
import matplotlib.image as iming
import bson.binary


if __name__ == '__main__':
    connect = MongoClient('127.0.0.1', 27017)
    db = connect.ShiTao
    collection = db.Seals

    imgput = gridfs.GridFS(db)
    dirs = 'D:\\Research\\Project\\data\\Seals\\'

    files = os.listdir(dirs)

    for file in files:
        filesname = dirs + file
        imgfile = iming.imread(filesname)

        f = file.split('.')
        datatmp = open(filesname, 'rb')
        data = BytesIO(datatmp.read())
        content = bson.binary.Binary(data.getvalue())
        insertimg = imgput.put(data, content_type=f[1], filename=f[0])
        # collection.update_one({'seal_name': file},
        #                      {'$set': {'seal_img': content}})
        datatmp.close()

    gridFS = gridfs.GridFS(db, collection="fs")
    count = 0
    for grid_out in gridFS.find():
        count += 1
        print(count)
        # data = grid_out.read()  # 获取图片数据
        # outf = open('%s.png' % count, 'wb')  # 创建文件
        # outf.write(data)  # 存储图片
        # print('ok')
        # outf.close()
