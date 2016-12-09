from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from nltk.corpus import stopwords

from sklearn.cluster import KMeans, MiniBatchKMeans


import sys, re
from time import time

import numpy as np
import codecs


catch = stopwords.words("english")
dataset= []

with codecs.open(sys.argv[1]+'/title_StackOverflow.txt', 'r', encoding='utf8') as f:
	for line in f:
		#line = re.sub('[^a-zA-Z]'," ", line)
		#line = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', line)
		#line = [word for word in line.split() if word not in catch]
		#line = ' '.join(line)
		dataset.append(line.lower())


vectorizer = TfidfVectorizer(max_df=0.5, max_features=5000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
X = vectorizer.fit_transform(dataset)
svd = TruncatedSVD(24)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)
print X.shape
explained_variance = svd.explained_variance_ratio_.sum()
true_k = 20

#if opts.minibatch:
#    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
#                        init_size=1000, batch_size=1000, verbose=opts.verbose)
#else:
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)


km.fit(X)
original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
tag = []

for i in range(true_k):
	ind = order_centroids[i, 0]
	tag.append(terms[ind])


print tag


test_data = np.genfromtxt(sys.argv[1]+'/check_index.csv',delimiter=',')

fout = open(sys.argv[2], 'w')
fout.write("ID,Ans\n")

label = np.zeros((20000,1))

for i in range(20000):
	for j in range(20):
		if dataset[i].find(tag[j]) != -1:
			label[i] = j+1
			
#print label
#exit()


for i in range(test_data.shape[0]-1):
	if label[test_data[i+1][1]] == 0:
		same = 0
	elif label[test_data[i+1][2]] == 0:
		same = 0
	else:
		same = 1 if label[test_data[i+1][1]] == label[test_data[i+1][2]] else 0	
	
	fout.write("%d,%d\n" % (i,same))










