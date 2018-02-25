from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import numpy as np
import tensorflow as tf
import argparse
import os.path as osp
from PIL import Image
from functools import partial

from eval import compute_map
#import models

import time
import matplotlib.pyplot as plt 

CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

debug_data_loader = 0


'''
def cnn_model_fn(features, labels, mode, num_classes=20):
	#input layer	
	input_layer = tf.resshape(features["x"], [-1,256,256,3])

	#conv layer1
	conv1 = tf.layers.conv2d(
		inputs= input_layer,
		filters = 32,
		kernel_size = [5,5],
		padding= "same"
		activation = tf.nn.relu)

	#pool1
	pool1 = tf.layers.max_pooling2d(
		inputs=conv1,
		pool_size =[2,2],
		strides=2)

	#conv2
	conv2 = tf.layers.conv2d(
		inputs = pool1,
		filters = 64,
		kernel_size = [5,5].
		padding = "same",
		activation = tf.nn.relu)

	#pooling layer 2
	pool1 = tf.layers.max_pooling2d(
		inputs = conv2,
		pool_size = [2,2],
		strides = 2)

	#dense layers
	pool2_flat = tf.reshape(pool2, [-1, 64 * 64 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=20)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        onehot_labels=labels, logits=logits), name='loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
'''


def load_pascal(data_dir, split ='train'):


	#loading data-list
	if split == "train":
		rel_path = "/ImageSets/Main/train.txt"
	elif split == "trainval":
		rel_path = "/ImageSets/Main/trainval.txt"
	elif split == "test":
		rel_path = "/ImageSets/Main/test.txt"
	data_list_file_path = data_dir + rel_path
	print('\n Location of', split,'data_list : ', data_list_file_path, '\n')
	data_list_file = open(data_list_file_path, 'r')
	data_list = data_list_file.readlines()
	data_list = [d.strip() for d in data_list if d.strip()]  #removing whitespaces and terminal characters
	data_list_file.close()

	
	#initializing data/labels/weight/ arrays
	number_of_images = len(data_list) #number of data points
	print(split ,'data size :', number_of_images, '\n')
	data = np.empty((number_of_images, 256,256,3), dtype = "float32")
	labels = np.empty((number_of_images, 20), dtype = "int")
	weights = np.empty((number_of_images, 20), dtype = "int")

	
	#loading images
	for i in range(number_of_images):
		
		rel_path = data_list[i] + '.jpg'
		path_img =  data_dir +'/JPEGImages/' + rel_path 
		if debug_data_loader == 1 :
			#path_img = data_dir + path_img
			print("data no :" , i, '\n')
			print('path to image :', path_img, '\n')

		
		img = Image.open(path_img)
		img = img.resize((256,256), Image.ANTIALIAS)
		
		data[i,:,:,:] = np.asarray(img, dtype = "float32")
		#print(data[i,:,20,2], '\n')
		#img.show()
		#time.sleep(1)
		img.close()
		del(img)

	print("Loaded ", split, "images")


	# loading weights and labels
	for il in range(len(CLASS_NAMES)):
		rel_path = CLASS_NAMES[il]+'_' +split +'.txt' 
		path_class = data_dir + '/ImageSets/Main/' + rel_path
		#print ('class:', CLASS_NAMES[il], '| path:', path_class, '\n')

		temp_l = []
		temp_w = []
		with open(path_class, 'r') as class_file:
			for row in class_file:
				temp_x, temp_y = row.split()
				if int(temp_y) == -1:
					temp_l = np.append(temp_l, int(0))
					temp_w = np.append(temp_w, int(1))
				elif int(temp_y) == 1:
					temp_l = np.append(temp_l, int(1))
					temp_w = np.append(temp_w, int(1))
				elif int(temp_y) == 0:
					temp_l = np.append(temp_l, int(1))
					temp_w = np.append(temp_w, int(0))

		class_file.close()
		if len(temp_l) != number_of_images or len(temp_w) != number_of_images :
			print("************** \n WARNING \n size of labels/weights and data doesn't match",
			" \n **************")

		labels[:,il] = temp_l
		weights[:, il] = temp_w
		del(temp_b, temp_a, temp_x,temp_y, temp_l, temp_w)
		if debug_data_loader == 1:
			print(labels[1:20,:])
			print(labels[number_of_images-20:number_of_images+100,:])
			print(labels[number_of_images-1,:])

	if debug_data_loader == 1:
		#print(np.size(labels))
		print(np.shape(weights))
		print(np.shape(labels))
		print(labels[0:10,:])
		print(weights[0:10,:])

	print("Loaded ", split, "weights and labels")

	return data, labels, weights 

	




def parse_args():
	parser = argparse.ArgumentParser(
		description='Train a classifier in tensorflow!')
	parser.add_argument(
		'data_dir', type=str, default='data/VOC2007',
		help='Path to PASCAL data storage')
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	args = parser.parse_args()

	print(args)
	return args


def main():
	args = parse_args()
	# Load training and eval data
	train_data, train_labels, train_weights = load_pascal(
		args.data_dir, split='trainval')
	#load_pascal(args.data_dir, split='trainval')
	eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')







if __name__ == "__main__":
   	main()
