from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#imports
import numpy as np 
import tensorflow as tf 

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
	""" Model function of CNN"""

	#input layer
	input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

	#conv layer1
	conv1 = tf.layers.conv2d(
		inputs = input_layer,
		filters =32,
		kernel_size =[5,5],
		padding = "same",
		activation = tf.nn.relu)

	#pooling layer 1
	pool1 =


if __name__ == "__main__":
	tf.app.run()


