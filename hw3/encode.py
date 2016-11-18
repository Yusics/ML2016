import numpy as np
import pickle
import os
import sys
import tensorflow as tf
from sklearn.cluster import KMeans as KM
from scipy.spatial.distance import euclidean as eu
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, normalization, Input, UpSampling2D
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras import backend as K

K.set_image_dim_ordering('th')


os.environ["THEANO_FLAGS"] = "device=gpu0"




class CNNModel(object):
	def __init__(self):
		self.class_num = 10
		self.epoch = 80


		train_data = pickle.load(open("data/all_label.p","rb"))
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
	

	def training(self):


		unlabel_raw = pickle.load(open(sys.argv[1]+"/all_unlabel.p","rb"))
		unlabel_raw = np.array(unlabel_raw)
		cl_r = np.reshape(unlabel_raw[:,0:1024], [45000, 1, 32, 32])
		cl_g = np.reshape(unlabel_raw[:,1024:2048], [45000, 1, 32, 32])
		cl_b = np.reshape(unlabel_raw[:,2048:3072], [45000, 1, 32, 32])
		self.unlabel_data = np.concatenate((cl_r, cl_g, cl_b), axis=1)
		self.unlabel_data =  self.unlabel_data.astype('float32')
		self.unlabel_data /= 255

		self.cnn_model = Sequential()

		self.unlabel_y = np.zeros((45000, 10))
		self.unlabel_x = np.zeros((1, 3, 32, 32))
		train_encode = self.encoder.predict(self.train_x)
		train_encode = np.array(train_encode)

		print "finish encoding"
		unlabel_encode = np.array(self.encoder.predict(self.unlabel_data))

		mean_train_encode = np.zeros((10,16,4,4))

		for i in range(10):
			mean_train_encode[i] = np.mean(train_encode[i*500:(i+1)*500],axis=0)
		loss = []

		for i in range(self.unlabel_data.shape[0]):
			#print "round %d" % i
			new_y = np.zeros((1,10))
			value = unlabel_encode[i].flatten()
			row_value = np.zeros(10)
			for j in range(10):
				row_value[j] = eu(value, mean_train_encode[j].flatten())

			self.unlabel_y[i][np.argmin(row_value)] = 1
			if np.min(row_value) > 3:
				loss.append(i)


		self.unlabel_data = np.delete(self.unlabel_data, np.s_[loss], 0)
		self.unlabel_y    = np.delete(self.unlabel_y, np.s_[loss], 0)

		print "finish label unlabeld_data"
		self.encoder.save_weights('auto_w', overwrite=True)

		self.train_x = np.concatenate((self.train_x, self.unlabel_data),axis=0)
		self.train_y = np.concatenate((self.train_y, self.unlabel_y), axis=0)


		self.cnn_model = Sequential()
		self.cnn_model.add(Convolution2D(16, 3, 3, input_shape=(3, 32, 32)))
		self.cnn_model.add(MaxPooling2D((2, 2)))
		self.cnn_model.add(Convolution2D(16, 3, 3))
		self.cnn_model.add(MaxPooling2D((2,2)))
		self.cnn_model.add(Convolution2D(16, 3, 3))
		self.cnn_model.add(MaxPooling2D((2, 2)))
		self.cnn_model.load_weights('auto_w')


		self.cnn_model.add(Flatten())

		self.cnn_model.add(Dense(output_dim = 689))
		self.cnn_model.add(Activation('sigmoid'))
		self.cnn_model.add(Dropout(0.25))

		self.cnn_model.add(Dense(output_dim = 10))
		self.cnn_model.add(Activation('softmax'))


		self.cnn_model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
		self.cnn_model.fit(self.train_x, self.train_y, batch_size=500, nb_epoch=1, shuffle=True)

		self.cnn_model.save(sys.argv[2])
		

	def autoencoderr(self):

		test_p = pickle.load(open(sys.argv[1]+"/test.p","rb"))
		test_data = np.array(test_p.get("data"))
		cl_r = np.reshape(test_data[:,0:1024], [10000, 1, 32, 32])
		cl_g = np.reshape(test_data[:,1024:2048], [10000, 1, 32, 32])
		cl_b = np.reshape(test_data[:,2048:3072], [10000, 1, 32, 32])
		cl = np.concatenate((cl_r, cl_g, cl_b), axis=1)
		cl = cl.astype('float32')
		cl /= 255


		self.unlabel_data = pickle.load(open(sys.argv[1]+"/unlabel_data", "rb"))
		self.unlabel_data =  self.unlabel_data.astype('float32')
		self.unlabel_data /= 255
		train_auto = np.concatenate((self.train_x, self.unlabel_data, cl))

		input_img = Input(shape=(3, 32, 32))


		x = Convolution2D(16, 3, 3, activation='relu', dim_ordering = "th", border_mode='same')(input_img)
		x = MaxPooling2D((2, 2), dim_ordering = "th", border_mode='same')(x)
		x = Convolution2D(16, 3, 3, activation='relu', dim_ordering = "th", border_mode='same')(x)
		x = MaxPooling2D((2, 2), dim_ordering = "th", border_mode='same')(x)
		x = Convolution2D(16, 3, 3, activation='relu', dim_ordering = "th", border_mode='same')(x)
		encoded = MaxPooling2D((2, 2), dim_ordering = "th", border_mode='same')(x)

		x = Convolution2D(16, 3, 3, activation='relu', dim_ordering = "th", border_mode='same')(encoded)
		x = UpSampling2D((2, 2), dim_ordering = "th")(x)
		x = Convolution2D(16, 3, 3, activation='relu', dim_ordering = "th", border_mode='same')(x)
		x = UpSampling2D((2, 2), dim_ordering = "th")(x)
		x = Convolution2D(16, 3, 3, activation='relu', dim_ordering = "th", border_mode='same')(x)
		x = UpSampling2D((2, 2), dim_ordering = "th")(x)
		decoded = Convolution2D(3, 3, 3, activation='sigmoid', dim_ordering = "th", border_mode='same')(x)


		adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		self.autoencoder = Model(input_img, decoded)
		self.autoencoder.summary()
		self.autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

		self.autoencoder.fit(train_auto, train_auto,
                nb_epoch=1,
                batch_size=1000,
                shuffle=True, validation_data=(cl,cl)
                )
		
			

		print "\nits encoder weight \n"
		
		self.encoder = Model(input = input_img, output = encoded)
		
		





			
def main():
	model = CNNModel()
	model.autoencoderr()
	model.training()
	

if __name__ == '__main__':
	main()



