import sys
import pickle
import string # punctuation
import numpy as np
from numpy import array
from string import digits # remove digit
from stop_words import get_stop_words
from wordsegment import segment
from compiler.ast import flatten
from numpy import genfromtxt



idx     = pickle.load(open("./model/test-id","rb"))
idx     = np.array(idx)
result  = pickle.load(open("result_new","rb"))


fout = open("result_pos.csv","w")
fout.write('"id","tags"\n')

for i in range(idx.shape[0]):
	fout.write('"%d","' % idx[i])
	if len(result[i]) > 0:
		try:	
			fout.write(result[i][0])
		except:
			fout.write("schroedinger-equation")
		for j in xrange(1,len(result[i]),1):
			try:
				fout.write(' %s' % result[i][j])
			except:
				fout.write(' %s' % "schroedinger-equation")

	fout.write('"\n')