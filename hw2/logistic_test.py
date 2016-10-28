#import pandas as pd
import numpy as np
import sys
import math
import pickle
#import matplotlib.pyplot as plt

class Model(object):

	def __init__(self):
		self.learning_rate  = 1e-1
		self.iteration      = 18000
		self.lamda          = 1e-5
		self.epsilon        = 1e-1
		self.feat_dim       = 57
		self.loss           = 0
		self.beta1          = 0.9
		self.beta2          = 0.999
		

	def process_data(self):
		self.raw_data = np.genfromtxt('spam_data/spam_train.csv',delimiter=',')
		self.size = self.raw_data.shape[0]
		
		self.x = self.raw_data[:, 1:58]
		
		#self.rm = np.s_[rm_feat]
		#self.feat_dim -= len(rm_feat)
		#self.x = np.delete(self.x, self.rm, 1)
		self.y = self.raw_data[:, 58]
		self.y = np.reshape(self.y, [self.y.shape[0], 1])
		self.w = np.random.rand(self.feat_dim, 1)
		self.b = np.random.rand(1,1)

	def sigmoid(self, z):
		return 1/(1+np.exp(-z/100)) 

	def cross_entropy(self):
		z = np.dot(self.x, self.w) + self.b
		loss = 0
		for i in range(self.y.shape[0]):
			if self.y[i][0] != self.sigmoid(z)[i][0]:
				loss += (-( self.y[i][0]*np.log(self.sigmoid(z)[i][0]) + (1-self.y[i][0])*np.log(1-self.sigmoid(z)[i][0]) ))
		return loss

	def gradient_decent(self):
		lr_t = self.learning_rate
		m_t_w  = np.zeros((self.feat_dim, 1))
		v_t_w  = np.zeros((self.feat_dim, 1))
		m_t_b  = 0
		v_t_b  = 0
		loss   = []
		for j in range(self.iteration):
			#loss = self.cross_entropy()
			z    = np.dot(self.x, self.w) + self.b
			grad_w = np.reshape(-np.sum((self.y - self.sigmoid(z))*self.x, axis=0), [self.feat_dim, 1]) + self.lamda*self.w
			grad_b = -np.sum(self.y - self.sigmoid(z))	
			if(j>0):
				lr_t = self.learning_rate * np.sqrt(1-self.beta2**j)/(1-self.beta1**j)
				m_t_w = self.beta1*m_t_w + (1-self.beta1) * grad_w
				v_t_w = self.beta2*v_t_w + (1-self.beta2) * (grad_w**2)
				m_t_b = self.beta1*m_t_b + (1-self.beta1) * grad_b
				v_t_b = self.beta2*v_t_b + (1-self.beta2) * (grad_b**2)
			self.w -=  lr_t * m_t_w/(np.sqrt(v_t_w) + self.epsilon)
			self.b -=  lr_t * m_t_b/(np.sqrt(v_t_b) + self.epsilon)

			#sys.stdout.write("\r training loss: %.3f" % loss)
			#sys.stdout.flush()
		loss_all = []
		for i in range(self.y.shape[0]):
			if self.y[i][0] != self.sigmoid(z)[i][0]:
				loss = (-( self.y[i][0]*np.log(self.sigmoid(z)[i][0]) + (1-self.y[i][0])*np.log(1-self.sigmoid(z)[i][0])))
				if loss > 1:
					loss_all.append(i)
		self.y = np.delete(self.y, np.s_[loss_all], 0)
		self.x = np.delete(self.x, np.s_[loss_all], 0)
		#print self.y.shape[0]
		#print self.x.shape[1]

		for j in range(self.iteration):
			#loss = self.cross_entropy()
			z    = np.dot(self.x, self.w) + self.b
			#print z
			grad_w = np.reshape(-sum((self.y - self.sigmoid(z))*self.x), [self.feat_dim, 1]) + self.lamda*self.w
			grad_b = -sum(self.y - self.sigmoid(z))	
			if(j>0):
				lr_t = self.learning_rate * np.sqrt(1-self.beta2**j)/(1-self.beta1**j)
				m_t_w = self.beta1*m_t_w + (1-self.beta1) * grad_w
				v_t_w = self.beta2*v_t_w + (1-self.beta2) * (grad_w**2)
				m_t_b = self.beta1*m_t_b + (1-self.beta1) * grad_b
				v_t_b = self.beta2*v_t_b + (1-self.beta2) * (grad_b**2)
			self.w -=  lr_t * m_t_w/(np.sqrt(v_t_w) + self.epsilon)
			self.b -=  lr_t * m_t_b/(np.sqrt(v_t_b) + self.epsilon)

	def classified(self, y):
		if y >= 0.5:
			return 1
		else:
			return 0
			

	def test(self):
		
		print "id,label"
		test_data  = np.genfromtxt(sys.argv[2], delimiter=',')
		test_feat = test_data[:, 1:58]
		#test_feat = np.delete(test_feat, self.rm, 1)
		y         = self.sigmoid(np.dot(test_feat, self.w)+self.b)
		
		print "1,%d" % self.classified(y[0])
		for i in xrange(1,y.shape[0],1):
			print "%d,%d" %(i+1, self.classified(y[i]))


def main():
	model = pickle.load(open( sys.argv[1], "rb" ) )
	#model.process_data()
	#model.w = tempM.w
	#model.b = tempM.b
	#model.gradient_decent()	
	#pickle.dump(model, open( "tmp.model", "wb" ) )
	model.test()

if __name__ == "__main__":
    main()


