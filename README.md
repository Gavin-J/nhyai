# nhyai 简介
基于深度学习的图像识别算法，海量大数据样本，准确识别图片和视频中的涉黄、 涉暴涉恐、政治敏感、民谣等内容，也能从美观和清晰等维度对图像进行筛选， 快速精准，解放审核人力。

演示站点： [https://ai.hn-ssc.com/sindex](https://ai.hn-ssc.com/sindex)

### 安装与使用
- Ubuntu18.04，Windows10虽然也能跑，但是环境整合会无比蛋疼
- python version >= 3.5.2
- pip install -r requirements.txt
- tensorflow,tensorflow-gpu version >= 1.13.1
- pip install torch torchvision
### 前端打包
- cd /frontend
- npm run build
### 关于caffe安装
- sudo apt install caffe-cpu 或 sudo apt install caffe-cuda                                 
- cp  -r /usr/lib/python3/dist-packages/caffe  /home/condas/envs/python3.6/lib/python3.6/

### 数据库割接：

- python backend/manage.py makemigrations

- python backend/manage.py migrate

### 后台运行：
- screen python backend/manage.py runserver 0.0.0.0:8000 --noreload --insecure

### 性能分析+后台运行:
- python backend/manage.py runprofileserver 0.0.0.0:8000 --noreload --insecure --kcachegrind --prof-path=/tmp/ai-profile-data

### 运行多任务:
- screen python backend/manage_task.py rqworker android ios default

##### 如遇到mysql安装错误，在ubuntu下执行如下命令：
- sudo apt-get install libmysqlclient-dev
- 创建用户Django：python backend/manage_task.py createsuperuser

# 项目截图
#### 暴恐色情识别
![暴恐色情识别](https://github.com/qihang2012/images/blob/master/imgs/ezgif-6-761b57feb74b.gif)

#### 通用识别
![通用识别](https://github.com/qihang2012/images/blob/master/imgs/ezgif-6-71cff4d50b97.gif)

#### 敏感文字过滤
![敏感文字过滤](https://github.com/qihang2012/images/blob/master/imgs/ezgif-6-e00e4f5d063d.gif)

#### 视频检测
![视频检测](https://github.com/qihang2012/images/blob/master/imgs/ezgif-6-1e8ce9a20323.gif)

### 参考
- https://github.com/kaldi-asr/kaldi
- https://github.com/yahoo/open_nsfw
- https://github.com/chineseocr/chineseocr
- https://github.com/eragonruan/text-detection-ctpn
