# **README**

##### **运行环境：**

- pycharm2021.3.3

- vue3+flask+mongodb

**安装依赖和代码运行：**

- 解压checkpoint放在process路径下，把model放在back/analyze路径下，解压venv，mongodb，front

- data放在back路径下

- ```python
  pip install -r requirements.txt
  ```

- 打开终端命令行开启mongodb服务

  ```python
  //windows
  net start mongodb
  
  //mac
  //use two terminals
  cd /Volumes/T7/File/Project/Project/mongodb/bin
  ./mongod --dbpath ../data/
  cd /Volumes/T7/File/Project/Project/mongodb/bin
  ./mongo
  ```

- 打开pycharm运行代码

  ```python
  cd front
  npm run serve
  cd back
  python app.py
  ```

- 浏览器打开localhost

**主要代码文件：**

-back

​	-analyze

​		-model-epoch-51.pth //VGG模型，用于印章整体特征提取

​		-network.py //VGG网络

​	-data

​		-classes //画册首页，用于作品总览瀑布流展示卡片

​		-dataset //数据集，用于VGG训练整体特征

​		-denoised //黑白印章去噪（未使用）

​		-extracted //印章前景

​		-images //书画作品原图

​		-processed //人工PS处理印章

​		-red //红色印章，用于交互式印章比对时计算重叠程度

​		-resized //224*224的印章原图

​		-seals //印章原图

​	-app.py //后端功能函数

-front

​	-public

​		-graphs //预处理的单个关系图谱和聚类图谱

​		-graph_all_class.json //印章总图谱

​	-src

​		-assets //网站前端图

​		-components

​			-analzepage.vue //印章自动化分析子页面

​			-compareseal.vue //交互比对分析页面

​			-detailpage.vue //作品详情页

​			-detectionpage.vue //作品总览页面

​			-helloworld.vue //网站首页

​			-horizontimeline.vue //时间轴子页面

​			-interactivecompare.vue //交互比对编辑框

​			-navipart.vue //导航栏组件

​			-relationgraph.vue //知识图谱子页面

-process

​	-checkpoint //VGG模型存储

​	-confidence.py //计算每个印章的关联特征和综合相似度并写入数据库

​	-data.py //VGG训练数据类

​	-dataset.py //制作印章数据集

​	-extract_seal.py //提取印章前景与制作红色印章，先去除印章黑色部分

​	-network.py //VGG网络

​	-similarity.py //计算单个印章的整体特征、局部特征、角点特征和轮廓特征，并根据四个特征的加权平均筛选相似印章

​	-train.py //训练特征提取模型

​	-vgg_features.py //使用训练好的模型提取印章的整体特征，保存在predict.txt中

**数据库设计**

- features //印章自动化分析结果存储

  seal_name //印章图像文件名

  seal_cls //训练时对应印章类别

  index_name //训练时对应印章文件名

  seal_interpretation //印章释文

  similar_cls //按相似度排序后的相似印章及两两印章对应的5个特征相似度

  confidence //印章综合相似度

  mean //前4个特征相似度加权平均

- images //原画册信息

  description //画册简介

  front_page //画册首页文件名

  image_name //该册页文件名

  name //画册名

  standard //该画册是否精品件

- relations //印章与原作品对应关系

  seal_name //印章图像文件名

  seal_path //印章图像存储路径

  seal_interpretation //印章释文

  processed_name //人工处理印章图像文件名

  processed_path //人工处理印章图像存储路径

  image_name //原作品文件名

  image_path //原作品图像存储路径

  front_page //原作品画册首页

  type //朱文或白文

  property //私人印或收藏印

  mark //是否标记为可疑印章

  class //印面类别

- seals //印章年代时间轴信息存储

  seal_name //印章图像文件名

  seal_interpretation //印章释文

  image_name //原作品文件名

  name //原作品画册名

  year //记载年份

**整体思路**

###### 印章赏析流程

从首页进入石涛作品总览，选择一个画册进入该画册作品详情页；作品详情页默认显示画册首页作品及其印章，点击可展示作品完整简介或切换至该画册的其余作品详情页；选择单个印章可对该印章进行自动化分析，根据印章的整体特征、局部特征、角点特征、轮廓特征和关联特征五个维度的加权平均得到该印章的综合相似度，并提供分析结论；提供了印章总图谱、关系图谱、聚类图谱和年代时间轴视图，便于专家结合作品中的收藏印、作品及印章之间的关系、年代等信息对印章进行更为全面的分析；在交互式印章比对页面中，提供了与自动排序的与该印章最相似的前20方印章，可在编辑框内对印章进行两两比对，计算两方印章的重叠程度。

###### 设计细节

- 首页动态展示石涛的作品《書畫》
- 作品详情页可对可疑印章做标记
- 收藏印的印章比对分析页面显示“该印章为收藏印”

###### 实现细节

- 印章检测：使用labelme对2568张网络上搜集的书画作品中的印章进行人工标注，制作印章检测数据集，其中包括旋转不同角度后的图像，以更好地适应扇形作品中的印章检测；使用Yolov4训练得到印章检测模型，并用该模型检测石涛作品中的印章
- 印章裁剪：对自动检测得到的印章进行校正，为避免印章检测不全的情况，裁剪时bounding box长款各扩大10pixel，并且只裁剪置信度大于95%以上的印章，由于特征提取使用的图像通常长款比为1:1，裁剪印章时以长边为基准裁剪成正方形，以保留原印章的空间比例特征
- 数据处理：考虑到很多印章会与题跋或背景图存在重叠现象，对裁剪得到的印章在RGB色彩空间按照阈值去除黑色部分，再用kmeans聚类的方法提取印章前景，便于后续进行特征提取
- 特征提取：将同一画册中相同印面的印章作为同一类，提取印章前景与书画作品中裁剪得到的背景图进行融合，构建新的印章，扩充数据集，用VGG训练印章特征提取模型，用于印章整体特征的提取；用sift算法提取印章的局部特征；harris算法提取印章的角点特征；用遍历找外轮廓并计算凸包的方法提取印章的轮廓特征；同一画册中其他石涛私人印的四维特征加权平均作为该印章的关联特征
- 相似度比对：整体特征相似度计算时对所有向量进行标准化后用两个向量的余弦作为相似度；局部特征相似度计算采用sift特征匹配个数与精品件印章的sift特征总数比值；角点特征相似度采用原印章角点特征个数与精品件印章角点特征个数比值；轮廓特征相似度采用两个印章轮廓线条的离散frechet距离来衡量