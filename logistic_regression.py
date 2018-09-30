import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#load the dataset
iris = load_iris()

#store all feature data in iris_x and target values in iris_y
iris_x, iris_y = iris.data[:], iris.target[:]
iris_y = pd.get_dummies(iris_y).values

#split train and test sets
trainX, testX, trainY, testY = train_test_split(iris_x, iris_y, test_size=0.33, random_state=42)

numFeatures = trainX.shape[1]
numLabels = trainY.shape[1]

#This will hold the features 
X = tf.placeholder(tf.float32, [None, numFeatures])

#This will be our correct answers matrix for 3 classes.
yGold = tf.placeholder(tf.float32, [None, numLabels])

#store random samples from a normal distribution with standard deviation 0.01
weights = tf.Variable(tf.random_normal([numFeatures, numLabels], mean=0, stddev=0.01, name="weights"))
bias = tf.Variable(tf.random_normal([1, numLabels], mean=0, stddev=0.01, name="bias"))

#define the equation of logistic regression y = sigmoid(w*x+b)
apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

#number of epochs in our training
numEpochs = 700

#define learning rate
learningRate = tf.train.exponential_decay(learning_rate=0.0008, global_step=1, decay_steps=trainX.shape[0], decay_rate=0.95, staircase=True)

#define cost function
cost_OP = tf.nn.l2_loss(activation_OP - yGold, name="squared_error_cost")

#define gradient descent
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

#create a tensorflow session
sess = tf.Session()

#initialize weights and biases
init_OP = tf.global_variables_initializer()
sess.run(init_OP)

#argmax(activation_OP, 1) returns the label with the most probability
#argmax(yGold, 1) is the correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP, 1), tf.argmax(yGold, 1))

#If every false prediction is 0 and every true prediction is 1, the average returns us the accuracy
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

# Initialize reporting variables
cost = 0
diff = 1
epoch_values = []
accuracy_values = []
cost_values = []

#Training epochs
for i in range(numEpochs):
	if i > 1 and diff < .0001:
		print("change in cost %g; convergence."%diff)
		break
	else:
		#Run training step
		step = sess.run(training_OP, feed_dict={X: trainX, yGold: trainY})
		#Report occasional stats
		if i % 10 == 0:
			#Add epoch to epoch_values
			epoch_values.append(i)
			#Generate accuracy stats on test data
			train_accuracy, newCost = sess.run([accuracy_OP, cost_OP], feed_dict={X: trainX, yGold: trainY})
			#add accuracy to live graphing variable
			accuracy_values.append(train_accuracy)
			#add cost to live graphing variable
			cost_values.append(newCost)
			#Re-assign values for variables
			diff = abs(newCost - cost)
			cost = newCost

			#generate print statements
			print("step %d, training accuracy %g, cost %g, change in cost %g" %(i, train_accuracy, newCost, diff))

#how well do we perform on held-out test data
print("final accuracy on test set: %s" %str(sess.run(accuracy_OP, feed_dict={X: testX, yGold: testY})))

#plot cost values
plt.plot([np.mean(cost_values[i-50:i]) for i in range(len(cost_values))])
plt.show()
