import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


DATA_FILE = 'data/birth_life_2010.txt'

## read in data from the xls file

def read_birth_life_data(filename):
    """
    Read in birth_life_2010.txt and return:
    data in the form of NumPy array
    n_samples: number of samples
    """
    text = open(filename, 'r').readlines()[1:]
    data = [line[:-1].split('\t') for line in text]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    data = list(zip(births, lifes))
    n_samples = len(data)
    data = np.asarray(data, dtype=np.float32)
    return data, n_samples

data, n_samples = read_birth_life_data(DATA_FILE)

## placeholders for input X and label Y
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

## weight and bias
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")

## model
Y_predicted = X * w + b

## loss function
loss = tf.square(Y - Y_predicted, name="loss")

## gradient descent with learning rate of 0.001 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
        ## initialize  variables
	sess.run(tf.global_variables_initializer()) 
	
	## train the model for 100 epochs
	for i in range(100):
                total_loss = 0
                for x, y in data:
                        # Session execute optimizer and fetch values of loss
                        _, l = sess.run([optimizer, loss], feed_dict={X: x, Y:y})
                        total_loss += l
                        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))
                        
	# Step 9: output the values of w and b
	w_out, b_out = sess.run([w, b]) 

# plot the results
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()
