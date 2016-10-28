#import pandas as pd
import numpy as np
import sys
import math
import pickle
#import matplotlib.pyplot as plt

class Model(object):

	def __init__(self):
		self.learning_rate  = 1e-3
		self.iteration      = 100000
		self.lamda          = 1e-3
		self.C              = 1
		self.epsilon        = 1e-8
		self.feat_dim       = 57
		self.loss           = 0
		self.beta1          = 0.9
		self.beta2          = 0.999
		

	def process_data(self):
		self.raw_data = np.genfromtxt(sys.argv[1],delimiter=',')
		self.size = self.raw_data.shape[0]
		self.x = self.raw_data[:, 1:58]
		
		
		#self.x[:, 55] = (self.x[:, 55]- np.mean(self.x[:, 55]))/np.std(self.x[:, 55])
		#self.x[:, 56] = (self.x[:, 55]- np.mean(self.x[:, 56]))/np.std(self.x[:, 56])

		self.y = [(lambda x: 1 if x==1 else -1)(x) for x in self.raw_data[:, 58]]
		self.y = np.reshape(self.y, [self.size])
		self.w = np.random.rand(self.feat_dim, 1)
		self.b = np.random.rand(1,1)

	def sigmoid(self, z):
		return 1/(1+np.exp(-z)) 

	def cross_entropy(self):
		z = np.dot(self.x, self.w) + self.b
		loss = 0
		for i in range(self.y.shape[0]):
			loss += self.C*max(0, 1-self.y[i]*z[i])
		loss += (1/2)*sum((self.w)**2)
		return loss

	

	def gradient_decent(self):
		lr_t = self.learning_rate
		m_t_w  = np.zeros((self.w.shape[0], 1))
		v_t_w  = np.zeros((self.w.shape[0], 1))
		m_t_b  = 0
		v_t_b  = 0
		loss   = []
		for j in range(self.iteration):
			#loss = self.cross_entropy()
			z    = np.dot(self.x, self.w) + self.b
			grad_w = 0
			grad_b = 0
			#grad_w = np.fromfunction(((lambda x: -self.y*self.x if x <1 else 0)(x) for x in (self.y*z)), (self.y.shape[0],self.x.shape[1]))
			#grad_b = np.fromfunction(((lambda x: -self.y if x <1 else 0)(x) for x in (self.y*z)), (self.y.shape[0],self.x.shape[1]))
			for i in range (self.y.shape[0]):

				if self.y[i]*z[i] <1:
					grad_w += np.sum(-self.y[i]*self.x[i,:])
					grad_b += (-self.y[i])

			grad_w += (self.lamda*self.w)
			#tmp_w = [(lambda x: -self.y*self.x if x <1 else 0)(x) for x in (self.y*z)]
			#tmp_b = [(lambda x: -self.y if x <1 else 0)(x) for x in (self.y*z)]
			#grad_w = np.reshape((sum(tmp_w)+self.lamda*self.w), [self.w.shape[0], self.w.shape[1]])
			#grad_b = sum(tmp_b)
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
			
	def classified(self, y):
		if y >= 0:
			return 1
		else:
			return 0
			

	def test(self):

		print "id,label"
		test_data  = np.genfromtxt(sys.arfv[1],delimiter=',')
		test_feat = test_data[:, 1:58]
		y         = np.dot(test_feat, self.w) + self.b
		
		print "1,%d" % self.classified(y[0])
		for i in xrange(1,y.shape[0],1):
			print "%d,%d" %(i+1, self.classified(y[i]))


def main():
	model = Model()
	#tempM = pickle.load(open( "tmp_svm.model", "rb" ) )
	model.process_data()
	#model.w = tempM.w
	#model.b = tempM.b
	model.gradient_decent()	
	pickle.dump(model, open(sys.argv[2], "wb" ) )
	#model.test()

if __name__ == "__main__":
    main()


