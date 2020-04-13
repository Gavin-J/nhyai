"""
名片
"""
from apphelper.image import union_rbox
import re
from .positions import positionClass


class businesscard:
    """
    名片结构化识别
    """

    def __init__(self, result):
        self.result = union_rbox(result, 0.2)
        self.N = len(self.result)
        self.res = {}
        self.business_name()
        self.position()
        self.company()
        self.address()
        self.email()
        self.phone()
        self.telephone()
        self.qq()
        self.webchat()

    def business_name(self):
        """
        姓名
        """
        business_name = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            res = re.findall("^([\u4E00-\u9FA5]+|[a-zA-Z]+)$",txt)
            if len(res) > 0:
                business_name['姓名'] = res[0]
                self.res.update(business_name)
                break

            if i == self.N-1 and len(res) <=0:
                business_name['姓名'] = '其他'
                self.res.update(business_name)
                break

    def position(self):
        """
        职位
        """
        position = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')  
            if positionClass().check_positionWords(txt.lower()):
                position['职位'] = txt
                self.res.update(position)
                break

    def company(self):
        """
        公司
        """
        company = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            if("公司" in txt):
                company['公司'] = txt
                self.res.update(company)
                break

    def address(self):
        """
        地址
        """
        address = {}
        addString=[]
        address_cx = 0
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')

            ##增加判断第二行地址X轴的偏移量不能大于200，否则视为其他信息
            cx = self.result[i]['cx']

            ##身份证地址
            if '地址' in txt  or  '省' in txt or '市' in txt or '县' in txt or '街' in txt or '村' in txt or "镇" in txt or "区" in txt or "城" in txt or "室" in txt or "房" in txt or "园" in txt:
                if address_cx !=0  and cx > address_cx - 200:
                    addString.append(txt.replace('地址:',''))
                
                if address_cx == 0:
                    addString.append(txt.replace('地址:',''))
                    address_cx = cx
            else:
                ##增加地址第二行判断
                res = re.findall(r'\d+号',txt)
            
            if len(res)>0:
                addString.append(txt.replace('地址:',''))
            
        if len(addString)>0:
            address['地址']  =''.join(addString)
            self.res.update(address) 
            

    def email(self):
        """
        邮箱
        """
        email = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            res = re.findall("(邮箱:[A-Za-z0-9]+)+(@+[a-zA-Z0-9_-]+\.)+([a-zA-Z0-9_-]+)$",txt)
            if len(res)>0:
                email['邮箱']  = res[0][0].replace('邮箱:','')+res[0][1]+res[0][2]
                self.res.update(email)
                break

    def phone(self):
        """
        手机
        """
        phone = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            res = re.findall("手机:[\u4E00-\u9FA5A-Za-z0-9]+",txt)
            if len(res)>0:
                phone['手机']  = res[0].replace('手机:','')
                self.res.update(phone)
                break
    
    def telephone(self):
        """
        电话
        """
        telephone = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            txt = txt.replace('电话:', '')
            res = re.findall("^((\d{7,8})|(\d{4}|\d{3})-(\d{7,8})|(\d{4}|\d{3})-(\d{7,8})-(\d{4}|\d{3}|\d{2}|\d{1})|(\d{7,8})-(\d{4}|\d{3}|\d{2}|\d{1}))$",txt)
            if len(res)>0:
                telephone['电话']  = res[0][0]
                self.res.update(telephone)
                break
    
    def qq(self):
        """
        QQ
        """
        qq = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            res = re.findall("QQ:[\u4E00-\u9FA5A-Za-z0-9]+",txt)
            if len(res)>0:
                qq['QQ']  = res[0].replace('QQ:','')
                self.res.update(qq)
                break
    
    def webchat(self):
        """
        微信
        """
        webchat = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            res = re.findall("微信:[\u4E00-\u9FA5A-Za-z0-9]+",txt)
            if len(res)>0:
                webchat['微信']  = res[0].replace('微信:','')
                self.res.update(webchat)
                break
