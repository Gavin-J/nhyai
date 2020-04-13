# usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import os
from django.conf import settings
from time import time

class positionClass:
    def __init__(self):
        self = self

    def check_positionWords_test(self, df, input_word):
        for index, row in df.iterrows():
            if row['职位'] == input_word:
                return True

            if index == df.iterrows() and row['职位'] != input_word:
                return False

    def check_positionWords(self, input_word):
        for index, row in settings.ZW.iterrows():
            if row['职位'] == input_word:
                return True

            if index == settings.ZW.iterrows() and row['职位'] != input_word:
                return False

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(os.getcwd(),"backend","api","ocr","application","position.csv"),encoding='gbk')
    input_word = "JAVA项目经理"
    start = time()
    flag = positionClass().check_positionWords_test(df, input_word.lower())
    totaltime = time() - start
    print ('cost: ' + str(totaltime))
    if flag == True:
        print (input_word)
    else:
        print ("未匹配到记录")
    
