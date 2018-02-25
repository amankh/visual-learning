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
import matplotlib.pyplot as plt

from eval import compute_map
#import models

tf.logging.set_verbosity(tf.logging.INFO)

debug_data_loader = 0 # set to 1 to debug using prints
test_cnn = 0# set to 1 to train/test only 20 images
test_cnn_batch = 50
BATCH_SIZE = 10
NUM_ITERS = 4

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


def cnn_model_fn(features, labels, mode, num_classes=20):
    '''
    ALEXNET ARCHITECTURE:
    -> image        [-1, 256, 256, 3]
    -> data augmentation [-1,224,224,3]

    -> conv(11, 4, 96, 'VALID')  [-1, 256-11/4 +1 , ]
    -> relu()
    -> max_pool(3, 2)
    -> conv(5, 1, 256, 'SAME')
    -> relu()
    -> max_pool(3, 2)
    -> conv(3, 1, 384, 'SAME')
    -> relu()
    -> conv(3, 1, 384, 'SAME')
    -> relu()
    -> conv(3, 1, 256, 'SAME')
    -> max_pool(3, 2)
    -> flatten()
    -> fully_connected(4096)
    -> relu()
    -> dropout(0.5)
    -> fully_connected(4096)
    -> relu()
    -> dropout(0.5)
    -> fully_connected(20)
    '''

    input_layer = tf.reshape(features["x"], [-1,256,256,3])
    weights = features["w"]


    #data augmentation


    #conv layer1
    conv1 = tf.layers.conv2d(
        inputs= input_layer,
        filters = 96,
        strides = 4,
        kernel_size = [11,11],
        padding= "valid",
        activation = tf.nn.relu)

    #pool1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size =[3,3],
        strides=2)

    #conv layer2
    conv2 = tf.layers.conv2d(
        inputs= pool1,
        filters = 256,
        strides = 1,
        kernel_size = [5,5],
        padding= "same",
        activation = tf.nn.relu)

    #pool2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size =[3,3],
        strides=2)

    #conv layer3
    conv3 = tf.layers.conv2d(
        inputs= pool2,
        filters = 384,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu)

    #conv layer4
    conv4 = tf.layers.conv2d(
        inputs= conv3,
        filters = 384,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu)
    
    #conv layer5
    conv5 = tf.layers.conv2d(
        inputs= conv4,
        filters = 256,
        strides = 1,
        kernel_size = [3,3],
        padding= "same",
        activation = tf.nn.relu)  ## check if activation is there or not


    #pool3
    pool3 = tf.layers.max_pooling2d(
        inputs=conv5,
        pool_size =[3,3],
        strides=2)




    
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def load_pascal(data_dir, split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 224px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
        weights: (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are confidently labeled and 0s for classes that 
            are ambiguous.
    """
    # Wrote this function
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
    if test_cnn ==1:
        number_of_images = test_cnn_batch

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

        labels[:,il] = temp_l[0:number_of_images]
        weights[:, il] = temp_w[0:number_of_images]
        del( temp_x,temp_y, temp_l, temp_w)
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
    return args


def _get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr


def main():
    args = parse_args()
    # Load training and eval data
    train_data, train_labels, train_weights = load_pascal(
        args.data_dir, split='trainval')
    eval_data, eval_labels, eval_weights = load_pascal(
        args.data_dir, split='test')

    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn,
                         num_classes=train_labels.shape[1]),
        model_dir="pascal_01")
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data, "w": train_weights},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)

    plt.figure(1)
    plt.ion()
    for it in range(4):
        print("SET :", it)
        pascal_classifier.train(
            input_fn=train_input_fn,
            steps=NUM_ITERS,
            hooks=[logging_hook])
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data, "w": eval_weights},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        #eval_results = pascal_classifier.evaluate(input_fn=eval_input_fn)
        #print(eval_results)
        pred = list(pascal_classifier.predict(input_fn=eval_input_fn))
        pred = np.stack([p['probabilities'] for p in pred])
        rand_AP = compute_map(
            eval_labels, np.random.random(eval_labels.shape),
            eval_weights, average=None)
        print('Random AP: {} mAP'.format(np.mean(rand_AP)))
        gt_AP = compute_map(
            eval_labels, eval_labels, eval_weights, average=None)
        print('GT AP: {} mAP'.format(np.mean(gt_AP)))
        AP = compute_map(eval_labels, pred, eval_weights, average=None)
        print('Obtained {} mAP'.format(np.mean(AP)))
        print('per class:')
        for cid, cname in enumerate(CLASS_NAMES):
            print('{}: {}'.format(cname, _get_el(AP, cid)))

        plt.plot(it, np.mean(AP), 'xr')
        plt.draw()
        plt.pause(0.0001)

    plt.ioff()





if __name__ == "__main__":
    main()