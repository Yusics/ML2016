# -*- coding: utf-8 -*-

import time
import sys
import gc

#Parameters

#Preprocessing========================================================
time_start = time.time()

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import re
import string
import pattern.en as pe

uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
def removeTagAndUri(x):
	soup = BeautifulSoup(x, "html.parser")
	if soup.code: soup.code.decompose()
	return re.sub(uri_re, "", soup.get_text())
	
def removePunc(x):
	x = x.lower()
	x = re.sub(r'[^\x00-\x7f]',r' ',x)
	x = re.sub(r'\\',r'',x)
	return re.sub("[" + string.punctuation + "]", " ", x)

stops = set(stopwords.words("english"))
def removeStopwords(x):
   filtered_words = [word for word in x.split() if word not in stops]
   return " ".join(filtered_words)

def removeShortWords(x):
	filtered_words = [word for word in x.split() if len(word) >= 3]
	return " ".join(filtered_words)
	
def removeNonNounAndStem(x):
	text = nltk.pos_tag(nltk.word_tokenize(x))
	new_text = []
	for word, pos in text:
		if ((pos == 'NN') or (pos == 'VBG') or (pos == 'NNS') or (pos == 'NNP') or (pos == 'NNPS')):
			gc.disable()
			new_text.append(word)
			gc.enable()
	return new_text

file_name = sys.argv[1] #========================================================
SCORING = False #===============================================
import csv
csv_file = open(file_name, 'r')

count = 0
data = []
for row in csv.DictReader(csv_file):
	article = {}
	
	article['id'] = row['id']
	
	title = removePunc(row['title'])
	title = removeStopwords(title)
	title = removeNonNounAndStem(title)
	article['title'] = title
	
	content = removeTagAndUri(row['content'])
	content = removePunc(content)
	content = removeStopwords(content)
	content = removeShortWords(content)
	content = removeNonNounAndStem(content)
	article['content'] = content

	if SCORING: article['tags'] = row['tags'].split()
	
	gc.disable()
	data.append(article)
	gc.enable()

	count += 1
	if (not count%1000): print 'Preprocessing... ' + str(count)
	if (count == 5000): break

csv_file.close()
print 'Preprocessing Done ' + str(time.time() - time_start)
	
#Topic phrase dectection and TF-IDF========================================
time_start = time.time()

DF_THRESHOLD = 3		#========================================================
KEYWORD_SIZE = 1000	#========================================================

def pushArticleToBag(list, bag):
	bag["article_num"] += 1
	for word in list:	
		bag["word_num"] += 1
		if (word not in bag["flag"]):
			if (word not in bag["dict"]):
				bag["dict"][word] = len(bag["dict"])
				gc.disable()
				bag["df"].append(1)
				bag["tf"].append(1)
				bag["list"].append(word)
				gc.enable()
				
			else:
				bag["df"][bag["dict"][word]] += 1
				bag["tf"][bag["dict"][word]] += 1
				
			bag["flag"][word] = 1
			
		else:
			bag["tf"][bag["dict"][word]] += 1
			
	bag["flag"] = {}

def bagMask(bag):
	for i in range(len(bag["list"])):
		if (bag["df"][i] < DF_THRESHOLD):
			bag["df"][i] = bag["article_num"]
	
import math as m
import numpy as np

def bagTFIDF(bag):
	newbag = [float(bag["tf"][i]) * m.log(float(bag["article_num"]) / float(bag["df"][i])) for i in range(len(bag["list"]))]
	return newbag
	
def bagIDF(bag):
	newbag = [m.log(float(bag["article_num"]) / float(bag["df"][i])) for i in range(len(bag["list"]))]
	return newbag
	
def topic_tfidf():
	bag_topic = {"dict" : {}, "list" : [], "df" : [], "tf" : [], "flag" : {}, "article_num" : 0, "word_num" : 0}

	for article in data:
		pushArticleToBag(article['title'] + article['content'], bag_topic)

	bagMask(bag_topic)
	bag_topic_idf_tfidf = zip(bag_topic["list"], bagIDF(bag_topic), bagTFIDF(bag_topic))
	bag_topic_idf_tfidf.sort(key = (lambda x: x[2]))
	bag_topic_idf_tfidf.reverse()

	keyword_list = [bag_topic_idf_tfidf[i][0] for i in range(KEYWORD_SIZE)]
	keyword_dict = dict([(keyword_list[i], i) for i in range(KEYWORD_SIZE)])
	
	return (bag_topic, bag_topic_idf_tfidf, keyword_dict, keyword_list)

def pushArticleToPhrase(list, bag):
	for i in range(len(list) - 1):
		if ((list[i] in keyword_dict) and (list[i+1] in keyword_dict)):
			index1 = keyword_dict[list[i]]
			index2 = keyword_dict[list[i+1]]
			bag[index1][0][index2] += 1
			bag[index1][1] += 1
		
PHRASE_RATIO = 0.2 #==============================================
def phraseJudge(bag):
	for i in range(KEYWORD_SIZE):
		for j in range(KEYWORD_SIZE):
			if bag[i][0][j] >= PHRASE_RATIO * bag[i][1]:
				bag[i][0][j] = 1
			else:
				bag[i][0][j] = 0

def phraseMerge(list, bag):
	if len(list) == 0: return

	new_list = []
	i = 0
	while i < (len(list) - 1):
		if ((list[i] in keyword_dict) and (list[i+1] in keyword_dict)):
			index1 = keyword_dict[list[i]]
			index2 = keyword_dict[list[i+1]]
			if bag[index1][0][index2] == 1:
				gc.disable()
				new_list.append(list[i] + '-' + list[i+1])
				gc.enable()
				i += 1
			else:
				gc.disable()
				new_list.append(list[i])
				gc.enable()
		i += 1
	if (i == len(list) - 1): new_list.append(list[-1])
	list[:] = new_list[:]

(bag_topic, bag_topic_idf_tfidf, keyword_dict, keyword_list) = topic_tfidf() #1st

bag_phrase = [[[0] * KEYWORD_SIZE, 0] for i in range(KEYWORD_SIZE)]
for article in data:
	pushArticleToPhrase(article['title'], bag_phrase)
	pushArticleToPhrase(article['content'], bag_phrase)
	
phraseJudge(bag_phrase)

for article in data:
	phraseMerge(article['title'], bag_phrase)
	phraseMerge(article['content'], bag_phrase)

(bag_topic, bag_topic_idf_tfidf, keyword_dict, keyword_list) = topic_tfidf() #2nd, after phrase detection
	
print 'Topic TF-IDF Done ' + str(time.time() - time_start)

#Article TF-IDF (Feature)========================================================
time_start = time.time()

def pushArticleToFeature(list):
	feature = [0.0] * KEYWORD_SIZE
	for word in list:
		if word in keyword_dict:
			feature[keyword_dict[word]] += 1		
	for i in range(KEYWORD_SIZE):
		feature[i] = feature[i] * bag_topic_idf_tfidf[i][1]
	return feature
		
Features = [np.array(pushArticleToFeature(article['title'] + article['content'])) for article in data]

print 'Article TF-IDF Done ' + str(time.time() - time_start)

#Topic Coherence List (Markov Matrix)========================================
time_start = time.time()

CoherenceList = np.zeros((KEYWORD_SIZE, KEYWORD_SIZE))
for feature in Features:
	CoherenceList += np.outer(feature, feature)
for i in range(len(CoherenceList.tolist())):
	if np.sum(CoherenceList[i]) == 0.: CoherenceList[i][i] = 1.
CoherenceList = np.array([np.array(row) / (np.sum(row)) for row in CoherenceList.tolist()])
CoherenceList = np.transpose(CoherenceList)

def markov(f):
	return np.matmul(CoherenceList, f)
	
def featuredWords(f):
	fw = zip(f, keyword_list)
	fw.sort(key = (lambda x: x[0]))
	fw.reverse()
	return fw

Features_2nd = [markov(feature) for feature in Features]
Features_3rd = [markov(feature) for feature in Features_2nd]

print 'Topic Coherence List Done ' + str(time.time() - time_start)
	
#Training / Predicting =========================================================
time_start = time.time()

Features_ranking		= [featuredWords(feature) for feature in Features]
Features_2nd_ranking	= [featuredWords(feature) for feature in Features_2nd]
Features_3rd_ranking	= [featuredWords(feature) for feature in Features_3rd]

#voting

def featureToRank(i):
	ranking_sum = [0] * KEYWORD_SIZE
	for j in range(KEYWORD_SIZE):
		index1 = keyword_dict[Features_ranking[i][j][1]]
		index2 = keyword_dict[Features_2nd_ranking[i][j][1]]
		index3 = keyword_dict[Features_3rd_ranking[i][j][1]]
		ranking_sum[index1] += j
		ranking_sum[index2] += j
		ranking_sum[index3] += j

	ranking_sum = zip(ranking_sum, keyword_list)
	ranking_sum.sort(key = (lambda x: x[0]))
	#if (not i%1000): print 'Training... ' + str(i)
	return ranking_sum

print 'Training / Predicting Done ' + str(time.time() - time_start)

#Output===============================================================
time_start = time.time()

TAG_SIZE = 2 #====================================================================
def rankToTags(i):
	tags = ''
	rank = featureToRank(i)
	for j in range(TAG_SIZE):
		tags += rank[j][1]
		if (j != TAG_SIZE - 1): tags += ' '

	return tags

csv_file = open('output.csv', 'w') #===============================================
fieldnames = ['id', 'tags']
writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n')
writer.writeheader()
for i in range(bag_topic["article_num"]):
	writer.writerow({'id': data[i]['id'], 'tags': rankToTags(i)})
	
csv_file.close()

print 'Output Done ' + str(time.time() - time_start)

def tagsToFeature(tags):
	feature = [0] * KEYWORD_SIZE
	for tag in tags:
		if tag in keyword_dict:
			index = keyword_dict[tag]
			feature[index] = 1
			
	return feature

def F1Score(a, b):
	if len(a) != len(b): print "F1Score error!"
	m = 0
	at = 0
	bt = 0
	for i in range(len(a)):
		if ((a[i] == 1) and (b[i] == 1)): m += 1
		if (a[i] == 1): at += 1
		if (b[i] == 1): bt += 1
	if ((at == 0) or (bt == 0)): return 0.
	p = float(m) / bt
	r = float(m) / at
	if (p + r == 0): return 0.
	return 2 * p * r / (p + r)

if SCORING:
	TotalF1Score = 0.
	for i in range(bag_topic["article_num"]):
		tag_predict = tagsToFeature(rankToTags(i).split())
		tag = tagsToFeature(data[i]['tags'])
		TotalF1Score += F1Score(tag, tag_predict) / bag_topic["article_num"]
	print "TotalF1Score: " + str(TotalF1Score) + ", Articles: " + str(bag_topic["article_num"])
