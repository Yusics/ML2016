import numpy as np
import pickle
import os
import sys
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, normalization
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

os.environ["THEANO_FLAGS"] = "device=gpu0"



class Model(object):
	def __init__(self):
		self.class_num = 10
		self.epoch = 80

		train_data = pickle.load(open(sys.argv[1]+"/all_label.p","rb"))
		train_data = np.array(train_data)

		self.train_y = np.zeros((5000,10))
		self.train_x = np.zeros((3,32,32))
		self.train_x = np.reshape(self.train_x, [1,3,32,32])
		for i in range(10):
			for j in range(500):
				self.train_y[i*500+j][i] = 1

				cl_r = np.reshape(train_data[i][j][0:1024],[1,32,32])
				cl_g = np.reshape(train_data[i][j][1024:2048],[1,32,32])
				cl_b = np.reshape(train_data[i][j][2048:3072],[1,32,32])
				cl = np.concatenate((cl_r, cl_g, cl_b), axis=0)
				cl = np.reshape(cl, [1,3,32,32])
				self.train_x = np.concatenate((self.train_x, cl),axis=0)

		self.train_x = np.delete(self.train_x, 0, 0)
	


	def training(self, epoch_num):
		self.cnn_model = Sequential()
		
		self.cnn_model.add(Convolution2D(30, 3, 3, dim_ordering = "th", input_shape=(3, 32, 32)))
		self.cnn_model.add(MaxPooling2D((2, 2), dim_ordering = "th"))

		self.cnn_model.add(Convolution2D(60, 3, 3, dim_ordering = "th"))
		self.cnn_model.add(MaxPooling2D((2, 2), dim_ordering = "th"))
		

		self.cnn_model.add(Flatten())

		self.cnn_model.add(Dense(output_dim = 689))
		self.cnn_model.add(Activation('sigmoid'))
		self.cnn_model.add(Dropout(0.25))


		self.cnn_model.summary()
		self.cnn_model.add(Dense(output_dim = 10))
		self.cnn_model.add(Activation('softmax'))

		self.cnn_model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
		self.cnn_model.fit(self.train_x, self.train_y, batch_size=500, nb_epoch=epoch_num, shuffle=True)

		

	def self_training(self):
		unlabel_raw = pickle.load(open(sys.argv[1]+"/all_unlabel.p","rb"))
		unlabel_raw = np.array(unlabel_raw)
		cl_r = np.reshape(unlabel_raw[:,0:1024], [45000, 1, 32, 32])
		cl_g = np.reshape(unlabel_raw[:,1024:2048], [45000, 1, 32, 32])
		cl_b = np.reshape(unlabel_raw[:,2048:3072], [45000, 1, 32, 32])
		unlabel_data = np.concatenate((cl_r, cl_g, cl_b), axis=1)
	
		
		loss = []
		self.unlabel_y = np.zeros((45000, 10))
		
		unlabel_result = self.cnn_model.predict(unlabel_data)
		for i in range(unlabel_data.shape[0]):
			self.unlabel_y[i][np.argmax(unlabel_result[i])]
			if max(unlabel_result[i]) < 0.9:
				loss.append(i)
			
		unlabel_data = np.delete(unlabel_data, np.s_[loss], 0)
		self.unlabel_y = np.delete(self.unlabel_y, np.s_[loss], 0)
		self.train_x = np.concatenate((self.train_x, unlabel_data))
		self.train_y = np.concatenate((self.train_y, self.unlabel_y))
			
			
		self.cnn_model = Sequential()

		
		
		self.cnn_model.add(Convolution2D(30, 3, 3, dim_ordering = "th", input_shape=(3, 32, 32)))
		self.cnn_model.add(MaxPooling2D((2, 2), dim_ordering = "th"))
		

		self.cnn_model.add(Convolution2D(60, 3, 3, dim_ordering = "th"))
		self.cnn_model.add(MaxPooling2D((2, 2), dim_ordering = "th"))

		self.cnn_model.add(Flatten())
		self.cnn_model.add(Dense(output_dim = 300))
		self.cnn_model.add(Activation('sigmoid'))
		self.cnn_model.add(Dropout(0.25))
		
		self.cnn_model.add(Dense(output_dim = 689))
		self.cnn_model.add(Activation('sigmoid'))
		self.cnn_model.add(Dropout(0.25))

		self.cnn_model.summary()
		self.cnn_model.add(Dense(output_dim = 10))
		self.cnn_model.add(Activation('softmax'))

		self.cnn_model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
		self.cnn_model.fit(self.train_x, self.train_y, batch_size=1000, nb_epoch=120, shuffle=True)

		self.cnn_model.save(sys.argv[2])

		






			
def main():
	model = Model()
	model.training(70)
	model.self_training()
	
	

if __name__ == '__main__':
	main()



