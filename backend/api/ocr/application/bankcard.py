"""
银行卡
"""
from apphelper.image import union_rbox
import re
from .banklist import banklist
from Pinyin2Hanzi import is_pinyin

class bankcard:
    """
    银行卡结构化识别
    """
    def __init__(self,result):
        self.result = union_rbox(result,0.2)
        self.non_decimal = re.compile(r'[^\d.]+')
        self.non_valid_decimal = re.compile(r'[^\d./]+')
        self.only_letter = re.compile("[^a-z^A-Z^]")
        self.N = len(self.result)
        self.res = {}
        self.bank_name()
        self.card_number()
        self.card_type()
        self.expiry_date()
        self.card_name()

    def bank_name(self):
        """
        银行名称
        """
        bank_name={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            res = re.findall("[\u4e00-\u9fa5]+",txt)
            if len(res)>0:
                for record in res:
                    bankName = banklist().get_bank_name(record)
                    if bankName is not None:
                        bank_name['银行名称']  = bankName
                        self.res.update(bank_name)
                        break

    def card_number(self):
        """
        卡号
        """
        card_number={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            res = txt
            txt = self.non_decimal.sub('', txt)
            if(txt.isnumeric() and len(txt)>15):
                card_number['卡号']  =txt
                self.res.update(card_number)
                break


    def card_type(self):
        """
        卡类型
        """
        card_type={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            res = txt
            txt = self.non_decimal.sub('', txt)
            if txt.isnumeric() and len(txt)==16:
                card_type['卡类型'] = '信用卡'
                self.res.update(card_type)
                break

            if txt.isnumeric() and len(txt) >16:
                card_type['卡类型'] = '借记卡'
                self.res.update(card_type)
                break


    def expiry_date(self):
        """
        有限期
        """
        expiry_date={}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ','')
            txt = txt.replace(' ','')
            res = txt
            txt = self.non_valid_decimal.sub('', txt)
            txt = txt[-5:]
            test_valid = re.search(r"\W",txt)
            if len(txt)==5 and test_valid and test_valid.group() == '/':
                month = txt.split('/')[0]
                year = '20' + txt.split('/')[1]
                expiry_date['有效期'] = year + '年' + month + '月'
                self.res.update(expiry_date)
                break


    def card_name(self):
        """
        卡名称
        """
        card_name={}
        for i in range(self.N):
            txt = self.result[i]['text']
            res = txt
            alpha = self.only_letter.sub('', txt)
            if alpha.isalpha() and len(alpha) > 6 and len(alpha) < 20 and \
                'card' not in alpha.lower()  and 'bank' not in alpha.lower() and \
                'month' not in alpha.lower():
                txt = self.only_letter.sub(' ', txt)
                #check firstname
                first_two = alpha.lower()[:2]
                first_three = alpha.lower()[:3]
                first_four = alpha.lower()[:4]
                first_five = alpha.lower()[:5]
                check_firstname = is_pinyin(first_two) or is_pinyin(first_three) or is_pinyin(first_four) or is_pinyin(first_five)
                if check_firstname:
                    card_name['卡名称']  = txt
                    self.res.update(card_name)
                    break

                names = txt.split(' ')
                for name in names:
                    name = name.replace(' ','').lower()
                    check_name = is_pinyin(name)
                    if check_name == True:
                        card_name['卡名称']  = txt
                        self.res.update(card_name)
                        break

