import numpy as np
import pickle
import sys
import os
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


'''config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))'''

os.environ["THEANO_FLAGS"] = "device=gpu0"




class CNNModel(object):
	def __init__(self):
		self.class_num = 10
		self.epoch = 80

		self.cnn_model = load_model(sys.argv[2])

	def testing(self):
		fout = open(sys.argv[3], 'w')
		fout.write("ID,class\n")

		test_p = pickle.load(open(sys.argv[1]+"/test.p","rb"))
		test_data = np.array(test_p.get("data"))
		cl_r = np.reshape(test_data[:,0:1024], [10000, 1, 32, 32])
		cl_g = np.reshape(test_data[:,1024:2048], [10000, 1, 32, 32])
		cl_b = np.reshape(test_data[:,2048:3072], [10000, 1, 32, 32])
		cl = np.concatenate((cl_r, cl_g, cl_b), axis=1)
		cl = cl.astype('float32')
		cl /= 255
		
		result = self.cnn_model.predict(cl)
		for i in range(10000):
			class_id = np.argmax(result[i])
			fout.write("%d,%d\n" % (i, class_id))






			
def main():
	
	model = CNNModel()
	model.testing()
	

if __name__ == '__main__':
	main()



