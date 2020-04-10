"""
行驶证
"""
from apphelper.image import union_rbox
import re
class vehiclelicense:
    """
    驾驶证结构化识别
    """
    def __init__(self,result):
        self.is_vehiclelicense = False
        self.is_vehicleplate = False
        self.result = union_rbox(result,0.2)
        self.N = len(self.result)
        self.res = {}
        self.license_type()
        self.plate_no()
        self.vehicle_type()
        self.owner()
        self.address()
        self.use_character()
        self.model()
        self.vin()
        self.engine_no()
        self.register_date()
        self.issue_date()

    def license_type(self):
        """
        类型 
        """
        license_type={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            res = re.findall("中华人民共和国机动车行驶证",txt)
            if len(res)>0:
                license_type['类型']  ='中华人民共和国机动车行驶证'
                self.res.update(license_type)
                self.is_vehiclelicense = True
                break 

            if i == self.N-1 and len(res) <=0:
                license_type['类型']  ='其他'
                self.res.update(license_type)
                break

    def plate_no(self):
        """
        号牌号码
        """
        plate_no={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')

            res = re.findall("[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}[A-Z0-9]{4}[A-Z0-9挂学警港澳]{1}",txt)
            if len(res) == 0:
                res = re.findall("号码[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}[A-Z0-9]{4}[A-Z0-9挂学警港澳]{1}",txt)

            if len(res)>0:
                plate_no['号牌号码']  = res[0].replace('号码','')
                self.res.update(plate_no) 
                self.is_vehicleplate = True
                break

    def vehicle_type(self):
        """
        车辆类型
        """
        vehicle_type={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            res = re.findall("[\u4E00-\u9FA5]+车$",txt)
            if len(res)>0:
                vehicle_type['车辆类型']  =res[0].replace('车辆类型','')
                self.res.update(vehicle_type)
                break 

    def owner(self):
        """
        所有人
        """
        owner={}
        res=[]
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')

            if self.is_vehiclelicense and txt != '中华人民共和国机动车行驶证':
                res = re.findall("人[\u4E00-\u9FA5]+",txt)
            if len(res)>0:
                owner['所有人']  =res[0].replace('人','')
                self.res.update(owner)
                break 

    def address(self):
        """
        住址
        """
        address={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            res = re.findall("址[\u4E00-\u9FA5\0-9]+",txt)
            if len(res)>0:
                address['住址']  =res[0].replace('址','')
                self.res.update(address)
                break


    def use_character(self):
        """
        使用性质
        """
        use_character={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            res = re.findall("[\u4E00-\u9FA5]{2,7}",txt)
            if '营运' in txt and len(res) > 0:
                use_character['使用性质']  =res[0].replace('使用性质','').replace('性质','')
                self.res.update(use_character)
                break

    def model(self):
        """
        品牌型号
        """
        model={}
        res = []
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            vehicleplate = re.findall("[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}[A-Z0-9]{4}[A-Z0-9挂学警港澳]{1}",txt)
            if self.is_vehicleplate == True and len(vehicleplate) > 0:
                self.is_vehicleplate = False

            res = re.findall("品牌型号[\u4E00-\u9FA5]+[A-Za-z0-9]+",txt)
            if len(res)>0:
                model["品牌型号"] = res[0].replace('品牌型号','').replace('非营运','').replace('使用性质','').replace('营运','').replace('性质','').replace('品脾型号','')
                self.res.update(model)
                break 

    def vin(self):
        """
        车辆识别代号
        """
        vin={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            res = re.findall("[A-Z\d]{15,17}$",txt)
            if len(res)>0:
                vin["车辆识别代号"] = res[0]
                self.res.update(vin)
                break


    def engine_no(self):
        """
        发动机号码
        """
        engine_no={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            res = re.findall("发动机号码[A-Za-z0-9]+",txt)
            if len(res) == 0:
                 res = re.findall("EngineNo.[A-Za-z0-9]+",txt)

            if len(res)>0:
                engine_no["发动机号码"] = res[0].replace('发动机号码','').replace('EngineNo.','')
                self.res.update(engine_no)
                break 

    def register_date(self):
        """
        注册日期
        """
        register_date={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            res = re.findall("[0-9]{1,4}\-[0-9]{1,2}\-[0-9]{1,2}",txt)
            if len(res)>0:
                register_date["注册日期"] = res[0]
                self.res.update(register_date)
                break

    def issue_date(self):
        """
        发证日期
        """
        issue_date={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            res = re.findall("[0-9]{1,4}\-[0-9]{1,2}\-[0-9]{1,2}$",txt)
            if len(res)>0:
                issue_date["发证日期"] = res[0]
                self.res.update(issue_date)
                break
