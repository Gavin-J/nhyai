# -*- coding: utf-8 -*-
"""
@author: lywen
"""
import os
import cv2
import json
import time
import uuid
import base64
from PIL import Image,ExifTags

# 添加当前项目到环境变量
import sys
sys.path.append(os.path.join(os.getcwd(),"backend","api","ocr"))
# print(sys.path)
from ocrmodel import model
from apphelper.image import union_rbox,adjust_box_to_origin,draw_boxes
from application import idcard,drivinglicense,vehiclelicense,businesslicense,bankcard,vehicleplate,businesscard
import numpy as np
from django.conf import settings

class OCR:
    """通用OCR识别、身份证识别"""
    def __init__(self):
        self = self

    def setExif(self, img_file):
           ##基于exif信息文字朝向检测
        img_exif = Image.open(img_file)
        img = cv2.imread(img_file)
        H,W = img.shape[:2]
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

        return img, is_exif,H,W

    def getTextList(self, img, angle, result):
        result = union_rbox(result,0.2)
        res = [{'text':x['text'],
                'name':str(i),
                'box':{'cx':x['cx'],
                        'cy':x['cy'],
                        'w':x['w'],
                        'h':x['h'],
                        'angle':x['degree']

                        }
                } for i,x in enumerate(result)]
        res = adjust_box_to_origin(np.copy(img), angle, res)##修正box
        textStrings = ''
        for each in res:
            textStrings += each["text"] + ' '
        return textStrings

    def getWordRecognition(self, img_file, bill_model):
        billModel = bill_model
        textAngle = True ##文字检测
        textLine = False ##只进行单行识别
        text = ''

        file_name = os.path.basename(img_file)
        file_path = os.path.dirname(img_file)

        # img = cv2.imread(img_file)##GBR
        img,is_exif,H,W = self.setExif(img_file)
        if is_exif is True:
            textAngle = False

        # H,W = img.shape[:2]
        timeTake = time.time()
        if textLine:
            ##单行识别
            partImg = Image.fromarray(img)
            text = model.crnnOcr(partImg.convert('L'))
            res =[ {'text':text,'name':'0','box':[0,0,W,0,W,H,0,H]} ]
        else:
            detectAngle = textAngle
            _,result,angle= model(img,
                                        detectAngle=detectAngle,##是否进行文字方向检测，通过web传参控制
                                        config=dict(MAX_HORIZONTAL_GAP=50,##字符之间的最大间隔，用于文本行的合并
                                        MIN_V_OVERLAPS=0.6,
                                        MIN_SIZE_SIM=0.6,
                                        TEXT_PROPOSALS_MIN_SCORE=0.1,
                                        TEXT_PROPOSALS_NMS_THRESH=0.3,
                                        TEXT_LINE_NMS_THRESH = 0.7,##文本行之间测iou值
                                                ),
                                        leftAdjust=True,##对检测的文本行进行向左延伸
                                        rightAdjust=True,##对检测的文本行进行向右延伸
                                        alph=0.01,##对检测的文本行进行向右、左延伸的倍数
                                       )



            if billModel=='' or billModel=='通用OCR' :
                text = self.getTextList(img,angle, result)
                result = union_rbox(result,0.2)
                res = [{'text':x['text'],
                        'name':str(i),
                        'box':{'cx':x['cx'],
                               'cy':x['cy'],
                               'w':x['w'],
                               'h':x['h'],
                               'angle':x['degree']

                              }
                       } for i,x in enumerate(result)]
                res = adjust_box_to_origin(np.copy(img),angle, res)##修正box
                com_res = res

            elif billModel=='身份证':

                res = idcard.idcard(result)
                res = res.res
                res =[ {'text':res[key],'name':key,'box':{}} for key in res]
                text = self.getTextList(img,angle, result)
                result = union_rbox(result,0.2)
                com_res = [{'text':x['text'],
                        'name':str(i),
                        'box':{'cx':x['cx'],
                               'cy':x['cy'],
                               'w':x['w'],
                               'h':x['h'],
                               'angle':x['degree']
                              }
                       } for i,x in enumerate(result)]
                com_res = adjust_box_to_origin(np.copy(img),angle, com_res)##修正box
            
            elif billModel=='驾驶证':

                res = drivinglicense.drivinglicense(result)
                res = res.res
                res =[ {'text':res[key],'name':key,'box':{}} for key in res]
                text = self.getTextList(img,angle, result)
                result = union_rbox(result,0.2)
                com_res = [{'text':x['text'],
                        'name':str(i),
                        'box':{'cx':x['cx'],
                               'cy':x['cy'],
                               'w':x['w'],
                               'h':x['h'],
                               'angle':x['degree']
                              }
                       } for i,x in enumerate(result)]
                com_res = adjust_box_to_origin(np.copy(img),angle, com_res)##修正box

            elif billModel=='行驶证':

                res = vehiclelicense.vehiclelicense(result)
                res = res.res
                res =[ {'text':res[key],'name':key,'box':{}} for key in res]
                text = self.getTextList(img,angle, result)
                result = union_rbox(result,0.2)
                com_res = [{'text':x['text'],
                        'name':str(i),
                        'box':{'cx':x['cx'],
                               'cy':x['cy'],
                               'w':x['w'],
                               'h':x['h'],
                               'angle':x['degree']
                              }
                       } for i,x in enumerate(result)]
                com_res = adjust_box_to_origin(np.copy(img),angle, com_res)##修正box

            elif billModel=='营业执照':

                res = businesslicense.businesslicense(result)
                res = res.res
                res =[ {'text':res[key],'name':key,'box':{}} for key in res]
                text = self.getTextList(img,angle, result)
                result = union_rbox(result,0.2)
                com_res = [{'text':x['text'],
                        'name':str(i),
                        'box':{'cx':x['cx'],
                               'cy':x['cy'],
                               'w':x['w'],
                               'h':x['h'],
                               'angle':x['degree']
                              }
                       } for i,x in enumerate(result)]
                com_res = adjust_box_to_origin(np.copy(img),angle, com_res)##修正box
        
            elif billModel=='银行卡':
                res = bankcard.bankcard(result)
                res = res.res
                res =[ {'text':res[key],'name':key,'box':{}} for key in res]
                text = self.getTextList(img,angle, result)
                result = union_rbox(result,0.2)
                com_res = [{'text':x['text'],
                        'name':str(i),
                        'box':{'cx':x['cx'],
                               'cy':x['cy'],
                               'w':x['w'],
                               'h':x['h'],
                               'angle':x['degree']
                              }
                       } for i,x in enumerate(result)]
                com_res = adjust_box_to_origin(np.copy(img),angle, com_res)##修正box

            elif billModel=='手写体':
                result = union_rbox(result,0.2)
                res = [{'text':x['text'],
                        'name':str(i),
                        'box':{'cx':x['cx'],
                               'cy':x['cy'],
                               'w':x['w'],
                               'h':x['h'],
                               'angle':x['degree']

                              }
                       } for i,x in enumerate(result)]
                res = adjust_box_to_origin(np.copy(img),angle, res)##修正box
                text = self.getTextList(img,angle, result)

            elif billModel=='车牌':
                res = vehicleplate.vehicleplate(result)
                res = res.res
                res =[ {'text':res[key],'name':key,'box':{}} for key in res]
                text = self.getTextList(img,angle, result)
                result = union_rbox(result,0.2)
                com_res = [{'text':x['text'],
                        'name':str(i),
                        'box':{'cx':x['cx'],
                               'cy':x['cy'],
                               'w':x['w'],
                               'h':x['h'],
                               'angle':x['degree']
                              }
                       } for i,x in enumerate(result)]
                com_res = adjust_box_to_origin(np.copy(img),angle, com_res)##修正box
            
            elif billModel=='名片':
                res = businesscard.businesscard(result)
                res = res.res
                res =[ {'text':res[key],'name':key,'box':{}} for key in res]
                text = self.getTextList(img,angle, result)
                result = union_rbox(result,0.2)
                com_res = [{'text':x['text'],
                        'name':str(i),
                        'box':{'cx':x['cx'],
                               'cy':x['cy'],
                               'w':x['w'],
                               'h':x['h'],
                               'angle':x['degree']
                              }
                       } for i,x in enumerate(result)]
                com_res = adjust_box_to_origin(np.copy(img),angle, com_res)##修正box
            
        
        timeTake = time.time()-timeTake

        #draw box in to original image
        drawBoxes = []
        draw_filename = file_name.split('.')[0] + '_drawed.' + file_name.split('.')[1]
        drawPath = os.path.join(file_path,draw_filename)
        drawUrl = ''
        # img =  cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if len(com_res) > 0:
            drawUrl = settings.FILE_URL +  settings.MEDIA_URL + 'photos' + '/' + draw_filename
            for arr in com_res:
               drawBoxes.append(arr["box"])
            drawImg = draw_boxes(np.array(img),drawBoxes)
            cv2.imwrite(drawPath, drawImg)

        return {'res':res,'timeTake':round(timeTake,4), 'text':text, 'com_res': com_res, 'drawUrl': drawUrl}

if __name__ == '__main__':
    ocrTest = OCR()
    idcard = ocrTest.getWordRecognition(os.path.join(os.getcwd(),"backend","api","ocr","test","idcard-demo.jpeg"),"身份证")
    print(idcard)