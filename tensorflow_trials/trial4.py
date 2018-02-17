import numpy as np
import tensorflow as tf

N, D, H = 64, 1000, 100

#setting up params
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))


init = tf.contrib.layers.xavier_initializer()
h = tf.layers.dense(inputs = x, units= H,
	activation = tf.nn.relu, kernel_initializer=init)
y_pred = tf.layers.dense(inputs = h, units= D, kernel_initializer= init)

loss = tf.losses.mean_squared_error(y_pred, y)

#backprop
optimizer = tf.train.GradientDescentOptimizer(1e-5)
updates = optimizer.minimize(loss)

#session to run the graph

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	values = {x:np.random.randn(N, D),
	y:np.random.randn(N, D),}

	for t in range(50):

		loss_val, _ = sess.run([loss, updates],
			feed_dict=values)







