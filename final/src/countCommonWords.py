from collections import Counter
from numpy import array
import numpy as np
import pickle
import matplotlib.pyplot as plt
from nltk.util import ngrams

class Model(object):
	def __init__(self):
		self.input = "test"
		self.common = []
		self.common2 = []
		self.th = 1200 # for 1gram
		self.th2 = 600 # for bigram
		#self.output = self.input
		'''self.id = []
		self.title = []
		self.content = []
		self.tags = []'''

	def load(self):
		self.title = pickle.load(open("model/" + self.input + "-title", "rb"))

	def count(self):
		text = []
		for i in range(len(self.title)):
			for j in range(len(self.title[i])):
				text.append(self.title[i][j])
		
		print "Text.len = ", len(text) 
		counts = Counter(text)

		temp = counts.most_common(self.th)
		for i in range(self.th):
			self.common.append(temp[i][0])
		print self.common[0]
		print "len = ", len(self.common)

		#bigram
		text = []
		for i in range(len(self.bigram)):
			for j in range(len(self.bigram[i])):
				text.append(self.bigram[i][j])
		
		print "Text-bigram.len = ", len(text) 
		counts = Counter(text)

		temp = counts.most_common(self.th2)
		for i in range(self.th2):
			self.common2.append(temp[i][0])
		print self.common2[0]
		print "len = ", len(self.common2)

	def capture(self):
		self.tags = []
		for i in range(len(self.title)):
			temp = [w for w in self.title[i] if w in self.common]
			temp2 = list(set(temp))
			#print temp2
			#self.tags.append(temp2)
			#for i in range(len(self.bigram)):
			temp = [w for w in self.bigram[i] if w in self.common2]
			temp22 = list(set(temp))
			self.tags.append(temp2+temp22)
		print "tags.len = ", len(self.tags)

	def loadId(self):
		self.id = pickle.load(open("model/" + self.input + "-id", "rb"))
		#print self.id[0]

	def submit(self):
		fout = open("result-" + self.input, 'w')
		fout.write('"id","tags"\n')
		for i in range(len(self.tags)):
			fout.write('"%d","' %self.id[i])
			l = len(self.tags[i])
			for j in range(l-1):
				fout.write("%s " %self.tags[i][j])
			if l > 0: 
				fout.write('%s"\n' %self.tags[i][l-1])
			else:
				fout.write('"\n')

	def ngrams(self):
		#print self.title[0]
		#self.title = [['1', '2', '3'], ['4', '5']]
		self.bigram = []
		for i in range(len(self.title)):
			temp2 = list(ngrams(self.title[i], 2))
			temp1 = []
			for j in range(len(temp2)):
				temp1.append(temp2[j][0] + '-' + temp2[j][1])
			self.bigram.append(temp1)
		#print self.bigram[:10]
		if len(self.title) != len(self.bigram):
			print "Warning!!!"

	def removeSame(self):
		c = 0
		for i in range(len(self.tags)):
			index = []
			for j in range(len(self.tags[i])-1): # remove the only word in the sequence
				for k in np.arange(j+1, len(self.tags[i])):
					if (self.tags[i][j] + "-" + self.tags[i][k]) in self.tags[i]:
						#print self.id[i],
						#print self.tags[i][j] + "-" + self.tags[i][k]
						index.append(self.tags[i][j])
						index.append(self.tags[i][k])
						c+=1
						
					elif (self.tags[i][k] + "-" + self.tags[i][j]) in self.tags[i]:
						#print self.id[i],
						#print self.tags[i][k] + "-" + self.tags[i][j]
						index.append(self.tags[i][j])
						index.append(self.tags[i][k])
						c+=1
		
			index = list(set(index))

			for j in range(len(index)):
				self.tags[i].remove(index[j])
		print c, len(self.tags)

model = Model()
model.load()
model.ngrams()
#print len(model.title), len(model.bigram)
#exit()
model.count()
model.capture()
model.loadId()
model.removeSame()
model.submit()


