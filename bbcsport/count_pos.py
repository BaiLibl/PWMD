# -*- coding: utf-8 -*-
from pyemd import emd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
from sklearn.cross_validation import train_test_split
from gensim.models import Word2Vec
from gensim.models import word2vec
import time

def func(array_in, array_flag):#取出距离矩阵内，每个文档所含有的单词的距离矩阵
    col_list = []
    num = array_flag.shape[0]
    for i in range(num):
        if array_flag[i] == 0:
            col_list.append(i)

    delete_array = np.delete(array_in, col_list, axis=0)
    delete_array = np.delete(delete_array, col_list, axis=1)
    return delete_array

def pos_distance(p1,p2,pos,P_,dat):
    p_1 = np.zeros(len(pos_word),dtype = np.double)
    p_2 = np.zeros(len(pos_word),dtype = np.double)
    p = {w:0 for w in pos_word}
    k =0
    for i in p1:
        w = i
        if pos[w] in p:
            p[pos[w]]+=1
            k+=1
    #print k
    for w in range(len(pos_word)):
        if pos_word[w] in p:
            p_1[w] = p[pos_word[w]] 
    p = {w:0 for w in pos_word}
    k =0
    for i in p2:
        w = i
        if pos[w] in p:
            p[pos[w]]+=1
            k+=1
    #print k
    for w in range(len(pos_word)):
        if pos_word[w] in p:
            p_2[w] = p[pos_word[w]] 
    #print p_1
    #print p_2
    p_1 = p_1.astype(np.double)
    p_2 = p_2.astype(np.double)
    if p_1.sum()!=0:
        p_11 =p_1/ p_1.sum()
    else:
        p_11 = p_1
    if p_2.sum()!=0:
        p_22 =p_2/ p_2.sum()
    else:
        p_22=p_2
    #dat = 1
    pos_score = dat*emd(p_11,p_22,P_)
    return pos_score

def get_rwmd_distance(v_1, v_2, D_,min_vocab=7, verbose=False):#d1和d2是计算的两个文本，min_vocab应该是两个文本中最少的词数量，计算两篇文档WMD距离
    #if(len(v1) < min_vocab or len(v2) < min_vocab):return emd(v_1,v_2,D_)
    dis1 = func(D_, v_1)
    dis2 = func(D_, v_2)
    v1 = [x  for x in v_1 if x > 0]
    v2 = [x  for x in v_2 if x > 0]
    v1 = np.array(v1)
    v2 = np.array(v2)
    if v1.sum()!=0:
        y1 = 1.0/v1.sum()
    else:
        y1=v1
    if v2.sum()!=0:
        y2 = 1.0/v2.sum()
    else:
        y2 = v2
    v1 = v1*y1
    v2 = v2*y2
    #if(len(v1) < min_vocab or len(v2) < min_vocab):return emd(v_1,v_2,D_)
    d1 = np.zeros(len(v1))
    k = 0
    for i in range(len(v_1)):
        if v_1[i] == 0:continue
        tmp = [D_[i][j] for j in range(len(v_2)) if v_2[j] > 0]
        d1[k] = min(tmp)
        k = k+1
    d2 = np.zeros(len(v2))
    k = 0
    for i in range(len(v_2)):
        if v_2[i] == 0:continue
        tmp = [D_[i][j] for j in range(len(v_1)) if v_1[j] > 0]
        d2[k] = min(tmp)
        k = k+1
    d12 = (v1*d1).sum()
    d21 = (v2*d2).sum()
    return max(d12,d21)
