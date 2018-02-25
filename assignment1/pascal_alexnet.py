'''
ARCHITECTURE:
    -> image
    -> conv(11, 4, 96, 'VALID')
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


def cnn_model_fn(features, labels, mode, num_classes=20):
    #input layer    
    input_layer = tf.resshape(features["x"], [-1,28,28,3])

    #conv layer1
    conv1 = tf.layers.conv2d(
        inputs= input_layer,
        filters = 96,
        strides = 4,
        kernel_size = [11,11],
        padding= "valid"
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
        padding= "same"
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
        padding= "same"
        activation = tf.nn.relu)

    #conv layer4
    conv4 = tf.layers.conv2d(
        inputs= conv3,
        filters = 384,
        strides = 1,
        kernel_size = [3,3],
        padding= "same"
        activation = tf.nn.relu)

    #pool2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size =[3,3],
        strides=2)


