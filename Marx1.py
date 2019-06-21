# -*- coding: utf-8 -*-
"""
Created on      2019-6-17 13:34 
把内容按照文档分开来统计
@author: zhouxianzheng
@Guosen Securities
"""


# -*- coding: utf-8 -*-
"""
Created on      2019-6-12 9:34 

@author: zhouxianzheng
@Guosen Securities
"""


import jieba.analyse
from io import open
import os
import pandas as pd
import time
import re

path = "/home/test/"  # 待读取的文件夹
path_list = os.listdir('C:/Users/guosen/Desktop/marx2/middle')
path_list.sort()  # 对读取的路径进行排序
marx_all = pd.Series(index=path_list)


for txtname in path_list:
    # txtname = '01a.txt'
    f = open('C:\\Users\\guosen\\Desktop\\marx2\\middle\\' + txtname, encoding='gb18030', errors='ignore')
    marx_all.loc[txtname] = f.readlines()
    f.close()

# 接下来，看看如何分词。


for j in range(len(path_list)):
    start = time.time()
    segments = []
    content = marx_all[path_list[j]]
    # len(content)
    # content[0]
    # content[100].split('\n')
    # content =["我来到北京清华大学,对读取的路径进行排序,对读取的路径进行排序我来到北京清华大学"]
    # 本PDF文件由S22PDF生成, S22PDF的版权由郭力所有pdf @ home.icm.ac.cn
    #TextRank 关键词抽取，只获取固定词性
    for i in range(len(content)):

        c = content[i].replace('本PDF文件由S22PDF生成, S22PDF的版权由郭力所有','')
        words = jieba.cut(c, cut_all=False)
        a = "#".join(words)
        b = a.split('#')
        splitedStr = ''
        for word in b:
            # 记录全局分词
            segments.append({'word':word, 'count':1})
            splitedStr += word + ' '
        # if i % 100 == 0:
        #     print(i)
    print(j)
    end = time.time()
    running_time = end-start
    print('time cost : %.5f sec' %running_time)

    dfSg = pd.DataFrame(segments)
    # 词频统计
    dfWord = dfSg.groupby('word')['count'].sum().sort_values(ascending=False)

    # 删除一个字的，删除标点符号，删除数字
    len(dfWord)
    dfWord = dfWord[[x.isalpha() for x in dfWord.index]]   #删除有标点符号的

    dfWord = dfWord[[len(x)>1 for x in dfWord.index]]  # 删除字符长度

    # pattern = re.compile('[0-9]+')
    # dfWord = dfWord[[pattern.findall(x).isdigit() for x in dfWord.index]]   #删除有标点符号的
    dfWord = dfWord[['\u4e00' <= x <= '\u9fa5' for x in dfWord.index]]   #删除有标点符号的
    # dfWord = dfWord[dfWord>=100]

    #导出csv
    dfWord.to_csv(u'D:\\package\\guosen\\marxengles results\\' + path_list[j][:3] + '.csv',encoding='gb18030')




# path_list[0][:3]









