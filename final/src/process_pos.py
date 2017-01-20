import sys
import pickle
import string # punctuation
import numpy as np
import nltk
import codecs
import itertools
from numpy import array
from string import digits # remove digit
from stop_words import get_stop_words
from wordsegment import segment
from compiler.ast import flatten
from numpy import genfromtxt

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans


predicted_tag = []

with codecs.open('result-test','r',encoding='utf-8') as f:
	for line in f:
		temp = line.strip().split(",")
		temp = temp[1].split(" ")
		temp[0] = temp[0].strip('"')
		temp[-1] = temp[-1].strip('"')
		predicted_tag.append(temp)

predicted_tag = predicted_tag[1:len(predicted_tag)]

all_tag = []

no_pos = []

with codecs.open('no.txt','r',encoding='utf-8') as f:
	for line in f:
		line = line.strip().split()
		no_pos.append(line[0])



content = pickle.load(open("./model/test-content","r"))
content = np.array(content)
idx     = pickle.load(open("./model/test-id","r"))
idx     = np.array(idx)
title   = pickle.load(open("./model/test-title", "r"))
title   = np.array(title)


pos_tags_title = []

for i in range(title.shape[0]):
	pp = nltk.pos_tag(title[i])
	pos_tags_title.append(pp)

#pickle.dump(pos_tags,open("pos_tags_title", "wb"))

pos_tags = []

for i in range(content.shape[0]):
	pp = nltk.pos_tag(content[i])
	pos_tags.append(pp)

#pickle.dump(pos_tags,open("pos_tags_title", "wb"))



#pos_tags = pickle.load(open("pos_tags", "r"))
pos_tags = np.array(pos_tags)


#pos_tags_title = pickle.load(open("pos_tags_title", "r"))
pos_tags_title = np.array(pos_tags_title)

new_tag = []

for i in range(title.shape[0]):
	tt = []
	for j in range(len(predicted_tag[i])):
		try:
			idx = content[i].index(predicted_tag[i][j])
			if pos_tags[i][idx][1] not in no_pos:
				if (predicted_tag[i][j] + 's') in all_tag:
					tt.append(predicted_tag[i][j]+'s')
				else:
					tt.append(predicted_tag[i][j])
		except:
			try:
				idx = title[i].index(predicted_tag[i][j])
				if pos_tags_title[i][idx][1] not in no_pos:
					if (predicted_tag[i][j] + 's') in all_tag:
						tt.append(predicted_tag[i][j]+'s')
					else:
						tt.append(predicted_tag[i][j])
			except:	
				
				if (predicted_tag[i][j] + 's') in all_tag:
					tt.append(predicted_tag[i][j]+'s')
				else:
					tt.append(predicted_tag[i][j])
	new_tag.append(tt)


pickle.dump(new_tag ,open("result_new","wb"))


		









