import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt


## define paramaters
learning_rate = 0.01
batch_size = 128
n_epochs = 30

# read in data
mnist = input_data.read_data_sets('data/mnist', one_hot=True)
X_batch, Y_batch = mnist.train.next_batch(batch_size)

## placeholders for input X and label Y
X = tf.placeholder(tf.float32, [batch_size, 784], name="image")
Y = tf.placeholder(tf.float32, [batch_size, 10], name="label")

## weight and bias
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weight")
b = tf.Variable(tf.zeros([1, 10]), name="bias")

## model
logits = tf.matmul(X, w) + b 

## loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy)

## gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

## calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())	
	n_batches = int(mnist.train.num_examples/batch_size)
	
	## train the model n_epochs times
	for i in range(n_epochs): 
		total_loss = 0

		for j in range(n_batches):
			X_batch, Y_batch = mnist.train.next_batch(batch_size)
			_, loss_batch = sess.run([optimizer, loss], {X: X_batch, Y:Y_batch}) 
			total_loss += loss_batch
		print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	## test the model
	n_batches = int(mnist.test.num_examples/batch_size)
	total_correct_preds = 0

	for i in range(n_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		accuracy_batch = sess.run(accuracy, {X: X_batch, Y:Y_batch})
		total_correct_preds += accuracy_batch	

	print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))
