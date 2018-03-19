# -*- coding: utf-8 -*-
from pyemd import emd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from knn3 import *
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
from sklearn.cross_validation import train_test_split
from gensim.models import Word2Vec
from gensim.models import word2vec
import time
import sys 


word_dis = np.loadtxt('one-hot-pos.txt')

filename = sys.argv[1]
fr = open(filename,'r')
line = fr.readline()
y = list()
while line:
    line = line.split()
    y.append(line[1])
    line = fr.readline()
fr.close()
#print len(y)

TRAIN_SIZE = int(len(y)*0.6)
TEST_SIZE = len(y)-TRAIN_SIZE
#TRAIN_SIZE = 4
#TEST_SIZE = 5
y = y[:TEST_SIZE+TRAIN_SIZE]
y_train = y[:TRAIN_SIZE] ##train
y_test  = y[TRAIN_SIZE:] ##test
print len(y),TRAIN_SIZE,TEST_SIZE

start = time.clock()

X_train = np.zeros((TRAIN_SIZE,2))
X_test = np.zeros((TEST_SIZE,2))
#print word_dis.shape
#print X_train.shape,len(y_train),X_test.shape,len(y_test)
knn_cv = WordMoversKNN(word_dis=word_dis, n_neighbors=10, n_jobs=1)
knn_cv.fit(X_train, y_train)
    
start = time.clock()
scr= knn_cv.score(X_test, y_test)
print "Test score: {:.3f}".format(scr)
#print 'test time used:',time.clock()-start

