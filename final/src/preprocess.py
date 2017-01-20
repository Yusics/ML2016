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

'''Undo:
output to model/file
add other preprocess and check
'''

class Model(object):
	def __init__(self):
		self.stopwords = get_stop_words('english')
		self.myStopwords = self.get_my_stop_words()
		self.inputList = ["biology", "crypto", "cooking", "diy", "robotics", "travel"]
		self.id = []
		self.title = []
		self.content = []
		self.tags = []

	def get_my_stop_words(self):
		f_stopwords = open("stop_words.txt", "r") 
		self.myStopwords = []  
		for line in f_stopwords: self.myStopwords.append(line[:-1])	
		f_stopwords.close()
		return self.myStopwords

	def preprocessSentence(self, l): # can adjust to be faster
		#l = l.replace("-", " ")
		#l = l.replace("_", " ")
		l = l.replace("<p>", " ")
		l = l.replace("</p>", " ")
		#l = l.replace(":", " ")
		l = l.translate(string.maketrans("",""), string.punctuation) # remove punctuation
		l = l.lower().translate(None, digits)
		l = [w for w in l.split() if (len(w) <= 24 and w[0:3] != "http")] # 20?
		#l = flatten([segment(w) for w in l])
		l = [w for w in l if w not in self.stopwords] # remove stopwords
		l = [w for w in l if w not in self.myStopwords]
		return l

	def load(self, filename):
		oneLine = [] #f = codecs.open(filename, 'r', encoding = 'utf8')
		f = open(filename, "r")
		next(f)
		for line in f:
			if line[0] == '"' and line[1].isdigit() == True:
				oneLine.append(line)
				#print "--", oneLine[-1]
			else:
				oneLine[-1] = oneLine[-1] + line
				#print "--", oneLine[-1]
			#if len(oneLine) > 10: break
		print "OneLine = ", len(oneLine)

		rawData = [] 
		for row in range(len(oneLine)):
			temp = 1
			rawData.append([])
			for w in range(len(oneLine[row])-3):
				if oneLine[row][w] == '"':
					if oneLine[row][w+1] == ',' and oneLine[row][w+2] == '"':
						rawData[row].append(oneLine[row][temp:w])
						temp = w+3
						#print "here", rawData[-1], len(rawData)
			if len(rawData[row])%4 == 3:
				rawData[row].append(oneLine[row][temp:-2])
				#print "here", rawData[-1], temp
			
			self.id.append(int(rawData[row][0]))  
			self.title.append(self.preprocessSentence(rawData[row][1]))
			self.content.append(self.preprocessSentence(rawData[row][2]))
			self.tags.append(rawData[row][3].split(" "))
			if row%5000 == 0: print row
		print "List = ", len(rawData)
		#rawData = array(rawData)
		#print "Array = ", rawData.shape
		f.close()

	def preprocessData(self):
		self.load("data/" + self.input + ".csv")
	
	def check(self):
		print "len = ", len(self.id), len(self.title), len(self.content), len(self.tags)
		#print self.title[0:5][:10]
		for i in np.arange(0, 100, 1):
			print "ID = ", model.id[i]
			print "Title = ", model.title[i]
			print "Content = ", model.content[i]
			print "Tags = ", model.tags[i]

	def check_tags(self):
		f = open("model/" + self.input + "-tags-check", "wb")
		for i in range(len(self.tags)):
			#print self.id[i], self.tags[i]
			f.write("%d, " %self.id[i])
			#print 
			for j in range(len(self.tags[i])):
				f.write("%s " %self.tags[i][j])
			f.write("\n")
			#"%d,%d\n" %(i, c1==c2)
		

	def save(self):
		pickle.dump(self.id, open("model/" + self.input + "-id", "wb"))
		pickle.dump(self.title, open("model/" + self.input + "-title", "wb"))
		pickle.dump(self.content, open("model/" + self.input + "-content", "wb"))
		pickle.dump(self.tags, open("model/" + self.input + "-tags", "wb"))

model = Model()
for i in range(6):
	model.input = model.inputList[i]
	model.preprocessData() # can command if wordVector and wordEmbedding are all zero
	#model.check()
	#model.check_tags()
	model.save()

