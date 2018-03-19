# -*- coding: utf-8 -*-
from pyemd import emd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.metrics import euclidean_distances
import os
import numpy as np
from gensim.models import Word2Vec
from gensim.models import word2vec
import time
import sys
 
from count_pos import *
TRAIN_SIZE = 4
TEST_SIZE = 5

fpos = sys.argv[1]
ftext = sys.argv[2]
fy = "label.txt"

#pos
pos_model = word2vec.Word2Vec.load("/root/wdk/gmm/pos/pos.model")
#pos_word = [word for word,vec in pos_model.wv.vocab.items()]
pos_word = ['PRP$', 'VBG', 'FW', 'VBN', 'VBP', 'WDT', 'JJ', 'WP', 'VBZ', 'DT', 'RP', 'NN', 'VBD', 'TO', 'PRP', 'RB', '-LRB-', 'NNS', 'LS', '``', 'WRB', 'CC', 'PDT', 'RBS', 'RBR', 'CD', 'EX', 'IN', 'WP$', 'MD', '-RRB-', 'JJS', 'JJR', 'SYM', 'VB', 'UH', 'NNP']
pos_D = np.array([pos_model[word] for word in pos_word])
pos_D.dtype=np.double
#pos_map
word_pos = dict()
fp = open(fpos,'r')
line = fp.readline()
while line:
    line = line.split()
    word_pos[line[0]] = line[1]
    line = fp.readline()
fp.close()

#doc ==>> pos
def doc2pos(d1,word_pos):
    d1 = d1.split()
    #print pos_word
    num = 0
    d = {w:0 for w in pos_word}
    for k in range(len(d1)):
        if d1[k] in word_pos.keys():
            p=word_pos[d1[k]]
            if p in pos_word:
                d[p]+=1
                num+=1
    #print [d[k] for k in d.keys()]
    dd = [(d[k]/float(num)) for k in d.keys()]
    dd = np.array(dd,dtype="float64")
    #print dd
    #dd = preprocessing.normalize(dd, norm='l1')
    return dd

filename = ftext
docs = list()
y = list()
cla = {}
fr = open(filename,'r')
k =1
for line in fr.readlines():
        if(len(line.split()) > 10):
                word = line.split()
                index = word[0]
                docs.append(line[len(word):])
                if(index in cla.keys()):
                        y.append(cla[index])
                else:
                        cla[index] = k
                        y.append(k)
                        k = k + 1
fr.close()
print(len(docs),len(y))

#SIZE = min(TRAIN_SIZE+TEST_SIZE,len(y))
SIZE = len(y)
#
vect = CountVectorizer(vocabulary=pos_word, dtype=np.double)
f = open(fy,'w')
for i in range(SIZE):
    f.write(str(i)+' '+str(y[i])+'\n')
f.close()

P_ = euclidean_distances(pos_D)
P_ = P_.astype(np.double)
P_/=P_.max()

TRAIN_SIZE = int(len(y)*0.6)
TEST_SIZE = len(y)-TEST_SIZE
'''
dis = np.zeros((TEST_SIZE,TRAIN_SIZE))
for k in range(TEST_SIZE):
    i = k+TRAIN_SIZE
    if k%50==0:print k
    d1 = doc2pos(docs[i],word_pos)
    #dis[i][i]=0
    j=0
    while j<TRAIN_SIZE:
        d2 = doc2pos(docs[j],word_pos)
        pos_dis = emd(d1,d2,P_)
        dis[k][j] = pos_dis 
        j+=1

print dis.shape
np.savetxt('pos_dis.txt',dis,fmt="%.5f")
'''




