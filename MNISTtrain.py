import tensorflow as tf
import numpy as np
import math
import time
import datetime
now = datetime.datetime.now()
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


import michele_binNN.networks as networks
import michele_binNN.input_data as input_data
import michele_binNN.optimizers as optimizers


##############################
### WE ARE TRAINING MINST  ###
##############################

def load_mnist():
	mnist = input_data.read_data_sets('dataset/MNIST_data', one_hot=False)
	x_train = mnist.train.images
	y_train = mnist.train.labels
	x_test = mnist.test.images
	y_test = mnist.test.labels
	return x_train, y_train, x_test, y_test, 10

# Type of network to be used: 2 choices
NETWORK = 'binary'
# NETWORK = 'binary_sbn'

# Dataset to be used for the learning task
DATASET = 'mnist'

# path where to save networks weights
MODELDIR = './models/'

# folder for tensorboard logs
LOGDIR = './logs/'

# Number of epochs performed during training
EPOCHS = 20

# Dimension of the training batch
BATCH_SIZE = 100

# Starting optimizer learning rate value
STEPSIZE = 1e-3

# Toggle th use of shift based AdaMax instead of vanilla Adam optimizer
SHIFT_OPT = False # if true use AdaMax

# CREATING AND DOWNLOADING DATA SESSION in folders

# timestamp = int(time.time()) # qua deve esserci data e ora
timestamp = now.strftime("%Y-%m-%d %H:%M")

model_name = ''.join([str(timestamp), '_', NETWORK, '_', DATASET])
session_logdir = os.path.join(LOGDIR, model_name)

train_logdir = os.path.join(session_logdir, 'train')
test_logdir = os.path.join(session_logdir, 'test')
session_modeldir = os.path.join(MODELDIR, model_name)

if not os.path.exists(session_modeldir):
	os.makedirs(session_modeldir)
if not os.path.exists(train_logdir):
	os.makedirs(train_logdir)
if not os.path.exists(test_logdir):
	os.makedirs(test_logdir)
	

	
# dataset preparation using tensorflow dataset iterators
x_train, y_train, x_test, y_test, num_classes = load_mnist()

batch_size = tf.placeholder(tf.int64)

# carico i dati di training
data_features, data_labels = tf.placeholder(tf.float32, (None,)+x_train.shape[1:]), tf.placeholder(tf.int32, (None,)+y_train.shape[1:])
train_data = tf.data.Dataset.from_tensor_slices((data_features, data_labels))
train_data = train_data.repeat().shuffle(x_train.shape[0]).batch(batch_size)

# carico dati test
test_data = tf.data.Dataset.from_tensor_slices((data_features, data_labels))
test_data = test_data.repeat().shuffle(x_test.shape[0]).batch(batch_size)

# itera sulla struttura dati
data_iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
features, labels = data_iterator.get_next()

# first step
train_initialization = data_iterator.make_initializer(train_data)
test_initialization = data_iterator.make_initializer(test_data)


# network initialization
is_training = tf.get_variable('is_training', initializer=tf.constant(False, tf.bool))
switch_training_inference = tf.assign(is_training, tf.logical_not(is_training))

# in this line you understand if using binary MLP or binShifted MLP
xnet, ynet = networks.get_network(NETWORK, DATASET, features, training=is_training)

with tf.name_scope('trainer_optimizer'):
	learning_rate = tf.Variable(STEPSIZE, name='learning_rate')
	learning_rate_decay = tf.placeholder(tf.float32, shape=(), name='lr_decay')
	update_learning_rate = tf.assign(learning_rate, learning_rate / learning_rate_decay)

	# here it's selected correct optimizer [ vanilla Adam / AdaMax ]
	opt_constructor = optimizers.ShiftBasedAdaMaxOptimizer if SHIFT_OPT else tf.train.AdamOptimizer
	optimizer = opt_constructor(learning_rate=learning_rate)

	# cross entropy is computed using sparse softmax function
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ynet, labels=labels)
	loss = tf.reduce_mean(cross_entropy)
	
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		global_step = tf.train.get_or_create_global_step()
		train_op = optimizer.minimize(loss=loss, global_step=global_step)

	
# metrics definition
with tf.variable_scope('metrics'):
	mloss, mloss_update	  = tf.metrics.mean(cross_entropy)
	accuracy, acc_update  = tf.metrics.accuracy(labels, tf.argmax(ynet, axis=1))

	metrics = [mloss, accuracy]
	metrics_update = [mloss_update, acc_update]

# Isolate the variables stored behind the scenes by the metric operation
metrics_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
metrics_initializer = tf.variables_initializer(metrics_variables)

# summaries
los_sum = tf.summary.scalar('loss', mloss)
acc_sum = tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge([los_sum, acc_sum])

# network weights saver
saver = tf.train.Saver()

NUM_BATCHES_TRAIN = math.ceil(x_train.shape[0] / BATCH_SIZE)
NUM_BATCHES_TEST = math.ceil(x_test.shape[0] / BATCH_SIZE)


#Plot settings

set_training_loss = []
set_training_acc = []
set_test_loss = []
set_test_acc = []
epoch_set=[]

with tf.Session() as sess:

	# tensorboard summary writer
	train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
	test_writer = tf.summary.FileWriter(test_logdir)
	
	sess.run(tf.global_variables_initializer())
	
	for epoch in range(EPOCHS):
		
		print("\nEPOCH %d/%d" % (epoch+1, EPOCHS))
		
		# exponential learning rate decay
		# ogni 10 epoche come nel paper

		if (epoch + 1) % 10 == 0:
			sess.run(update_learning_rate, feed_dict={learning_rate_decay: 2.0})
		
		
		# initialize training dataset and set batch normalization training
		sess.run(train_initialization, feed_dict={data_features:x_train, data_labels:y_train, batch_size:BATCH_SIZE})
		sess.run(metrics_initializer)
		sess.run(switch_training_inference)
		
		print("TRAINING ")
		# Training of the network
		for current_batch in tqdm(range(NUM_BATCHES_TRAIN)):
			sess.run(train_op)	# train network on a single batch
			batch_trn_loss, _ = sess.run(metrics_update)
			training_loss, training_accuracy = sess.run(metrics)

		print("")
		print("Training loss: ",round(training_loss,3), " Training accuracy: ", round(training_accuracy,3))
		print("")

		summary = sess.run(merged_summary)
		train_writer.add_summary(summary, epoch)

		# initialize the test dataset and set batch normalization inference
		sess.run(test_initialization, feed_dict={data_features:x_test, data_labels:y_test, batch_size:BATCH_SIZE})
		sess.run(metrics_initializer)

		# switch between training phase and test phase
		sess.run(switch_training_inference)

		# evaluation of the network
		for current_batch in tqdm(range(NUM_BATCHES_TEST)):
			sess.run([loss, metrics_update])
			val_loss, val_acc = sess.run(metrics)

		print("")
		print("Test loss:",round(val_loss,3)," Test accuracy: ",round(val_acc,3))
		print("")

		summary  = sess.run(merged_summary)
		test_writer.add_summary(summary, epoch)

		#-------------------------------------------------
		# inserisco nel grafico
		epoch_set.append(epoch+1)

		set_training_loss.append(training_loss)
		set_training_acc.append(training_accuracy)

		set_test_loss.append(val_loss)
		set_test_acc.append(val_acc)
		#-------------------------------------------------

		train_writer.close()
		test_writer.close()

		saver.save(sess, os.path.join(session_modeldir, 'model.ckpt'))

	plt.figure(1)
	# figure 1.1 is about loss variance in time for training and test
	plt.subplot(211)
	plt.plot(epoch_set, set_training_loss, 'b', label = 'trainig')
	plt.plot(epoch_set, set_test_loss, 'r', label='test')
	plt.legend()
	plt.xlabel('epochs')
	plt.ylabel('LOSS')

	# figure 1.2 is about accuracy variance in time for training and test
	plt.subplot(212)
	plt.plot(epoch_set, set_training_acc, 'b', label='training')
	plt.plot(epoch_set, set_test_acc, 'r', label='test')
	plt.legend()
	plt.xlabel('epochs')
	plt.ylabel('ACCURACY')

	plt.show()


print('\nTraining completed!\nNetwork model is saved in  {}\nTraining logs are saved in {}'.format(session_modeldir, session_logdir))