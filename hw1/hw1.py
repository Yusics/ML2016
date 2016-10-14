#import pandas as pd
import numpy as np
import sys
import math
#import matplotlib.pyplot as plt

class Model(object):

	def __init__(self):
		self.learning_rate  = 1e-3
		self.iteration      = 30000
		self.lamda          = 1
		self.epsilon        = 1e-8
		self.hour           = 9
		self.dim            = 9
		self.loss           = 0
		self.beta1          = 0.9
		self.beta2          = 0.999

	def process_data(self):
		self.raw_data = np.genfromtxt('data/train.csv',delimiter=',')
		self.raw_data = np.delete(self.raw_data, 0, 0)
		size = self.raw_data.shape[0]
		self.row = self.raw_data[:18, 3:27]
		self.x = self.raw_data[:18, 3:27]
		self.x = np.delete(self.x, [1,4,5,10,11,13,15,16,17], 0)
		day = size/18
		
		for i in range(day):

			row    = self.raw_data[i*18:(i+1)*18, 3:27]
			self.row = np.concatenate((self.row, row), axis=1)
			row    = np.delete(row, [1,4,5,10,11,13,15,16,17], 0)

			self.x = np.concatenate((self.x, row), axis=1)
		self.x = np.delete(self.x, np.s_[0:24], 1)
		self.row = np.delete(self.row, np.s_[0:24], 1)
		
		'''for i in xrange(day-50, day, 1):
			row    = self.raw_data[i*18:(i+1)*18, 3:self.raw_data.shape[1]-1]
			row    = np.delete(row, 10, 0)
		 	self.x = np.concatenate((self.x, row), axis=1)'''


		for i in range(self.x.shape[0]):
			for j in range(self.x.shape[1]):
				if math.isnan(self.x[i][j]):
					self.x[i][j] = 0

		self.x_para = np.reshape(self.x[:, 0:9], (self.hour * self.dim, 1))
		self.y    = self.raw_data[9, 12]

		
		for j in range(12):
			for i in xrange(9, 480, 1):
				data_x = np.reshape(self.x[:, j*480+i-9:j*480+i], (self.hour * self.dim, 1))
				self.x_para = np.concatenate((self.x_para, data_x), axis=1)

				self.y      = np.append(self.y, self.row[9, j*480+i]) 
		

		self.x_para = np.delete(self.x_para, [0], 1)
		self.y      = np.delete(self.y, [0], 0)

		self.w    = np.random.rand(1, self.dim * self.hour)
		self.b    = np.random.rand(1,1)



	def gradient_decent(self):
		lr_t = self.learning_rate 
		m_t_w  = np.zeros((self.w.shape[0], self.w.shape[1]))
		v_t_w  = np.zeros((self.w.shape[0], self.w.shape[1]))
		m_t_b  = 0
		v_t_b  = 0
		loss   = []
		
		for j in range(self.iteration):
			self.loss = 0
			grad_w = np.zeros((self.w.shape[0], self.w.shape[1]))
			grad_b = 0
			for i in range(self.y.shape[0]):
				grad_w += -2 * np.sum(self.y[i] - (np.dot(self.w, self.x_para[:,i]) + self.b)) * self.x_para[:, i] + 2 * self.lamda* self.w
				grad_b += -2 * np.sum(self.y[i] - (np.dot(self.w, self.x_para[:,i]) + self.b))
				self.loss += (self.y[i] - (np.dot(self.w, self.x_para[:,i]) + self.b))**2 + self.lamda*(np.sum(self.w**2))

			
			#loss.append(self.loss)
			if(j>0):
				lr_t = self.learning_rate * np.sqrt(1-self.beta2**j)/(1-self.beta1**j)
				m_t_w = self.beta1*m_t_w + (1-self.beta1) * grad_w
				v_t_w = self.beta2*v_t_w + (1-self.beta2) * (grad_w**2)
				m_t_b = self.beta1*m_t_b + (1-self.beta1) * grad_b
				v_t_b = self.beta2*v_t_b + (1-self.beta2) * (grad_b**2)

			self.w -=  lr_t * m_t_w/(np.sqrt(v_t_w) + self.epsilon)
			self.b -=  lr_t * m_t_b/(np.sqrt(v_t_b) + self.epsilon)
			#g = np.sum(grad_w ** 2) * np.ones((self.w.shape[0], self.w.shape[1]))
			#g_b = grad_b**2
			#print self.loss
			#sys.stdout.write("\r training loss: %.3f" % self.loss)
			#sys.stdout.flush()

		for i in range(self.y.shape[0]):
			if abs(self.y[i] - (np.dot(self.w, self.x_para[:,i]) + self.b)) > 11:
				 loss.append(i)

		self.y = np.delete(self.y, np.s_[loss], 0)
		self.x_para = np.delete(self.x_para, np.s_[loss], 1)
		print self.y.shape

		for j in range(self.iteration):
			self.loss = 0
			grad_w = np.zeros((self.w.shape[0], self.w.shape[1]))
			grad_b = 0
			for i in range(self.y.shape[0]):
				grad_w += -2 * np.sum(self.y[i] - (np.dot(self.w, self.x_para[:,i]) + self.b)) * self.x_para[:, i] + 2 * self.lamda* self.w
				grad_b += -2 * np.sum(self.y[i] - (np.dot(self.w, self.x_para[:,i]) + self.b))
				self.loss += (self.y[i] - (np.dot(self.w, self.x_para[:,i]) + self.b))**2 + self.lamda*(np.sum(self.w**2))

			
			#loss.append(self.loss)
			if(j>0):
				lr_t = self.learning_rate * np.sqrt(1-self.beta2**j)/(1-self.beta1**j)
				m_t_w = self.beta1*m_t_w + (1-self.beta1) * grad_w
				v_t_w = self.beta2*v_t_w + (1-self.beta2) * (grad_w**2)
				m_t_b = self.beta1*m_t_b + (1-self.beta1) * grad_b
				v_t_b = self.beta2*v_t_b + (1-self.beta2) * (grad_b**2)

			self.w -=  lr_t * m_t_w/(np.sqrt(v_t_w) + self.epsilon)
			self.b -=  lr_t * m_t_b/(np.sqrt(v_t_b) + self.epsilon)

	
	'''def test(self):
		self.test_loss = 0
		for i in xrange(self.y.shape[0]-50, self.y.shape[0]):
			self.test_loss += abs(self.y[i] - (np.dot(self.w, self.x_para[:,i]) + self.b))
			print "y: ",self.y[i],"   predict: ",(np.dot(self.w, self.x_para[:,i]) + self.b)
		self.test_loss = self.test_loss/50
		print "test_loss: ", self.test_loss'''

	def test(self):

		print "id,value"
		test_data = np.genfromtxt('data/test_X.csv',delimiter=',')
		test_x    = test_data[:18, 2:11]
		test_x    = np.delete(test_x, [1,4,5,10,11,13,15,16,17], 0)
		'''for i in range(test_x.shape[0]):
			for j in range(test_x.shape[1]):
				if math.isnan(test_x[i][j]):
					test_x[i][j] = -10'''
		

		test_x_para = np.reshape(test_x, (self.dim * self.hour, 1))
		print "id_0,%.4f" % np.dot(self.w, test_x_para)
		day       = test_data.shape[0]/18
		for i in xrange(1, day, 1):
			test_x = test_data[i*18:(i+1)*18, 2:11]
			test_x    = np.delete(test_x, [1,4,5,10,11,13,15,16,17], 0)
			'''for l in range(test_x.shape[0]):
				for n in range(test_x.shape[1]):
					if math.isnan(test_x[l][n]):
						test_x[l][n] = -10'''
			
			test_x_para = np.reshape(test_x, (self.dim * self.hour, 1))
			print "id_%d,%.4f" %(i,np.dot(self.w, test_x_para))


def main():
	model = Model()
	model.process_data()
	model.gradient_decent()	
	model.test()

if __name__ == "__main__":
    main()


