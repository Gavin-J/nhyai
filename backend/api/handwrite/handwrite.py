# -*- coding: utf-8 -*-
"""
@author: wangshujing
"""
import os
import cv2
import numpy as np

# 添加当前项目到环境变量
import sys
sys.path.append(os.path.join(os.getcwd(),"backend","api","handwrite"))
from dnn.main import text_ocr
from hwconfig import scale,maxScale,TEXT_LINE_SCORE,adjust
from ocr.config import AngleModelPb,AngleModelPbtxt
from PIL import Image,ExifTags
from api.util import rectifyImgAngle,changeImgAngle
class HandWrite:
    """手写体识别"""
    def __init__(self):
        self = self
        ##dnn 文字方向检测
        self.angleNet = cv2.dnn.readNetFromTensorflow(AngleModelPb,AngleModelPbtxt)

    def angle_detect_dnn(self, img,adjust=True):
        """
        文字方向检测
        """
        h,w = img.shape[:2]
        ROTATE = [0,90,180,270]
        if adjust:
            thesh = 0.05
            xmin,ymin,xmax,ymax = int(thesh*w),int(thesh*h),w-int(thesh*w),h-int(thesh*h)
            img = img[ymin:ymax,xmin:xmax]##剪切图片边缘
        
        
        inputBlob = cv2.dnn.blobFromImage(img, 
                                        scalefactor=1.0, 
                                        size=(224, 224),
                                        swapRB=True ,
                                        mean=[103.939,116.779,123.68],crop=False);
        self.angleNet.setInput(inputBlob)
        pred = self.angleNet.forward()
        index = np.argmax(pred,axis=1)[0]
        return ROTATE[index]

    def getWord(self, img_file):
        ##基于exif信息文字朝向检测
        img_exif = Image.open(img_file)
        file_name = os.path.basename(img_file)
        file_path = os.path.dirname(img_file)
        img = cv2.imread(img_file)
        is_exif = False
        try:
            for orientation in ExifTags.TAGS.keys() : 
                if ExifTags.TAGS[orientation]=='Orientation' : break 
            exif = dict(img_exif._getexif().items())
            if  exif[orientation] == 3 :
                img = Image.fromarray(img).transpose(Image.ROTATE_180)
                is_exif = True
            elif exif[orientation] == 8 : 
                img = Image.fromarray(img).transpose(Image.ROTATE_270)
                is_exif = True
            elif exif[orientation] == 6 : 
                img = Image.fromarray(img).transpose(Image.ROTATE_90)
                is_exif = True
        except:
            pass
        
        # print (img_file)
        # print (img)
        angle = 0
        if img is not None:
            ##基于tensorflow文字朝向检测
            if adjust and is_exif is False:
                angle = self.angle_detect_dnn(img=np.copy(img))
                if angle==90:
                    img = Image.fromarray(img).transpose(Image.ROTATE_270)
                elif angle==180:
                    img = Image.fromarray(img).transpose(Image.ROTATE_180)
                elif angle==270:
                    img = Image.fromarray(img).transpose(Image.ROTATE_90)
           
           #霍夫矫正
            if is_exif is False:
                angle = rectifyImgAngle(img_file)
                img = cv2.imread(img_file)

            image = np.array(img)
            # image =  cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            data, drawUrl = text_ocr(image,scale,maxScale,TEXT_LINE_SCORE, file_path, file_name)

            res = {'data':data,'errCode':0, 'drawUrl': drawUrl,'angle':angle}
        else:
            res = {'data':[],'errCode':3, 'drawUrl': '','angle':angle}
        return res


if __name__ == '__main__':
    handWriteTest = HandWrite()
    handword = handWriteTest.getWord(os.path.join(os.getcwd(),"backend","api","handwrite","test","img.jpeg"))
