# -*- coding: utf-8 -*-
from pyemd import emd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.metrics import euclidean_distances
from sklearn.feature_extraction import stop_words
import os
import numpy as np
from gensim.models import Word2Vec
from gensim.models import word2vec
import time
import sys
import gensim

from count_pos import *
TRAIN_SIZE = 4
TEST_SIZE = 5
ftext = sys.argv[1]
beta = float(sys.argv[2])
fy = 'label.txt'
fpos = 'r8_pos.txt'
#model = Word2Vec.load("/root/wdk/gmm/word2vec/word_model.mod")
model = gensim.models.KeyedVectors.load_word2vec_format('/root/wdk/wmd_representation/word2vec/GoogleNews-vectors-negative300.bin',binary=True)
W = np.array([model[w] for w in model.vocab])
vocab_dict = {w: k for k, w in enumerate(model.vocab)}
pos_word = ['PRP$', 'VBG', 'FW', 'VBN', 'VBP', 'WDT', 'JJ', 'WP', 'VBZ', 'DT', 'RP', 'NN', 'VBD', 'TO', 'PRP', 'RB', '-LRB-', 'NNS', 'LS', '``', 'WRB', 'CC', 'PDT', 'RBS', 'RBR', 'CD', 'EX', 'IN', 'WP$', 'MD', '-RRB-', 'JJS', 'JJR', 'SYM', 'VB', 'UH', 'NNP','NL']
pos_dict = dict()
for i in range(len(pos_word)):
    a = np.zeros(len(pos_word),dtype=float)
    #print a.shape
    a[i]=1.0
    pos_dict[pos_word[i]]=a
pos_dict['NL']= np.zeros(len(pos_word),dtype=float)

#pos_map
word_pos = dict()
fp = open(fpos,'r')
line = fp.readline()
while line:
    line = line.split()
    word_pos[line[0]] = line[1]
    line = fp.readline()
fp.close()

def doc2pos(d1,word_pos):
    d1 = d1.split()
    num = 0
    d = {w:0 for w in pos_word}
    for k in range(len(d1)):
        if d1[k] in word_pos.keys():
            p=word_pos[d1[k]]
            d[p]+=1
            num+=1
    dd = [(d[k]/float(num)) for k in d.keys()]
    dd = np.array(dd,dtype="float64")
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
#print SIZE
SIZE = len(y)
TRAIN_SIZE = int(len(y)*0.6)
TEST_SIZE = len(y)-TRAIN_SIZE
print TRAIN_SIZE,TEST_SIZE

f = open(fy,'w')
for i in range(SIZE):
    f.write(str(i)+' '+str(y[i])+'\n')
f.close()
#
dis = np.zeros((TEST_SIZE,TRAIN_SIZE))
for k in range(TEST_SIZE):
    if k%50==0:print k
    #print docs[i]
    i = k+TRAIN_SIZE
    d1 = docs[i]
    j = 0
    while j<TRAIN_SIZE:
        d2 = docs[j]
        vect = CountVectorizer(stop_words="english").fit([d1, d2])
        common = [w for w in set(d1.lower().split() + d2.lower().split()) if w in model.vocab and w not in stop_words.ENGLISH_STOP_WORDS]
        vect = CountVectorizer(vocabulary=common, dtype=np.double)
        v_1, v_2 = vect.transform([d1, d2])
        v_1 = v_1.toarray().ravel()
        v_2 = v_2.toarray().ravel()
        W_ = W[[vocab_dict[w] for w in vect.get_feature_names()]]
        D_ = euclidean_distances(W_)
        word = [w for w in vect.get_feature_names()]
        pp = [word_pos.get(w,'NL') for w in vect.get_feature_names()]
        P_ = [pos_dict[p] for p in pp]
        P_ = euclidean_distances(P_)
        P_ /=P_.max()
       	C = P_*D_
        D_ =D_+beta*C
        v_1 = v_1.astype(np.double)
        v_2 = v_2.astype(np.double)
        v_11 =v_1/ v_1.sum()
        v_22 =v_2/ v_2.sum()
        D_ = D_.astype(np.double)
        D_ /= D_.max() 
        word_dis = get_rwmd_distance(v_1, v_2, D_)
        #word_dis = 1
        dis[k][j] = word_dis
        j=j+1

print dis.shape
#save txt
np.savetxt('one-hot-pos.txt',dis,fmt='%.5f')



