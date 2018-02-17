import numpy as np
import tensorflow as tf

N, D, H = 64, 1000, 100

#setting up params
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))
w1 = tf.Variable(tf.random_normal(D, H)) # marked as trainable by default
w2 = tf.Variable(tf.random_normal(H, D)

#forward pass
h = tf.maximum(tf.matmul(x, w1), 0)  #relu
y_pred = tf.matmul(h,w2)
#diff = y_pred - y
#loss = tf.reduce_mean(tf.reduce_sum(diff**2, axis =1))
loss = tf.losses.mean_squared_error(y_pred, y)

#backprop
optimizer = tf.train.GradientDescentOptmizer(1e-5)
updates = optimizer.minimize(loss)

#session to run the graph

with tf.Session() as sess:
	sess.run(tf.global_variable_initializer())
	values = {x:np.random.randn(N, D),
	y:np.random.randn(N, D),}

	for t in rane(50):

		loss_val, _ = sess.run([loss, updates],
			feed_dict=values)







