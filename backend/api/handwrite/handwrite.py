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
from hwconfig import scale,maxScale,TEXT_LINE_SCORE
from ocr.config import AngleModelPb,AngleModelPbtxt
from PIL import Image

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
        img = cv2.imread(img_file)##GBR
        # print (img_file)
        # print (img)
        if img is not None:
            ##文字朝向检测
            angle = self.angle_detect_dnn(img=np.copy(img))
            if angle==90:
                img = Image.fromarray(img).transpose(Image.ROTATE_90)
            elif angle==180:
                img = Image.fromarray(img).transpose(Image.ROTATE_180)
            elif angle==270:
                img = Image.fromarray(img).transpose(Image.ROTATE_270)

            image = np.array(img)
            image =  cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            data = text_ocr(image,scale,maxScale,TEXT_LINE_SCORE)

            res = {'data':data,'errCode':0}
        else:
            res = {'data':[],'errCode':3}
        return res


if __name__ == '__main__':
    handWriteTest = HandWrite()
    handword = handWriteTest.getWord(os.path.join(os.getcwd(),"backend","api","handwrite","test","img.jpeg"))
