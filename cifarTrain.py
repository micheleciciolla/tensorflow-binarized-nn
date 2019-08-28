__author__ = "Flavio Lorenzi"
# AIRO Neural Network Project, Sapienza University
# updated 25 August


import tensorflow as tf
import numpy as np
import math
import datetime
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import networks
import optimizers

now = datetime.datetime.now()
##############################
## WE ARE TRAINING CIFAR-10 ##
##############################

#LOAD CIFAR10
#https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data
def load_cifar10():
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	return x_train, np.squeeze(y_train), x_test, np.squeeze(y_test), 10

#PARAMETERS INTERFACE

DATASET = 'cifar10'
NETWORK = 'binary'  #can choice between binary and binary_sbn
EPOCHS = 20
LR = 1e-3     #Optimizer Learning Rate
LOGDIR = './logs/'
MODELDIR = './models/'
BATCHSIZE = 100
SWITCH = True # if true we pass on SHIFT BASED ADAMAX from the standard VANILLA ADAM



#DOWNLOADING DATA SESSION in folders
timestamp = now.strftime("%Y-%m-%d %H:%M")  #per salvarla con data e ora

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
	
# CIFAR10 UPLOADING using tensorflow dataset iterators
x_train, y_train, x_test, y_test, num_classes = load_cifar10()

batch_size = tf.placeholder(tf.int64)

#carico i dati di training
data_features, data_labels = tf.placeholder(tf.float32, (None,)+x_train.shape[1:]), tf.placeholder(tf.int32, (None,)+y_train.shape[1:])
train_data = tf.data.Dataset.from_tensor_slices((data_features, data_labels))
train_data = train_data.repeat().shuffle(x_train.shape[0]).batch(batch_size)

#carico dati test
test_data = tf.data.Dataset.from_tensor_slices((data_features, data_labels))
test_data = test_data.repeat().shuffle(x_test.shape[0]).batch(batch_size)

#itera sulla struttura dati
data_iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
features, labels = data_iterator.get_next()

#first step
train_initialization = data_iterator.make_initializer(train_data)
test_initialization = data_iterator.make_initializer(test_data)



# NETWORK INIT
#If training is true, upload the network
is_training = tf.get_variable('is_training', initializer=tf.constant(False, tf.bool))
switch_training_inference = tf.assign(is_training, tf.logical_not(is_training))
xnet, ynet = networks.get_network(NETWORK, DATASET, features, training=is_training) 

with tf.name_scope('trainer_optimizer'):
	
	learning_rate = tf.Variable(LR, name='learning_rate')
	
	#Ad ogni iter il learning rate decade e va aggiornato
	learning_rate_decay = tf.placeholder(tf.float32, shape=(), name='lr_decay')
	update_learning_rate = tf.assign(learning_rate, learning_rate / learning_rate_decay)
	


	#Ottimizzatore corrente

	#Adam is an optimization algorithm that can used instead of the classical stochastic gradient descent procedure
	#Shift Based Adamax is an extension of Adam:
	opt_constructor = optimizers.ShiftBasedAdaMaxOptimizer if SWITCH  else tf.train.AdamOptimizer
	optimizer = opt_constructor(learning_rate=learning_rate)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ynet, labels=labels)

	# loss function: "ci fa capire quanto funziona la rete"
    # reduce_mean : calcola la media degli elementi (tra le dimensioni di un tensore)
	loss = tf.reduce_mean(cross_entropy)
	
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		global_step = tf.train.get_or_create_global_step()

		#def operazione di training
		train_op = optimizer.minimize(loss=loss, global_step=global_step)

	

# metrics utilizzato per le due funzioni utili: mean e accuracy
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
merged_summary = tf.summary.merge([los_sum, acc_sum])   #utile per training e evaluation


# network weights saver
saver = tf.train.Saver()

NUM_BATCHES_TRAIN = math.ceil(x_train.shape[0] / BATCHSIZE)
NUM_BATCHES_TEST = math.ceil(x_test.shape[0] / BATCHSIZE)

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
	

	#STAMPA
	for epoch in range(EPOCHS):
		
		print("\nEPOCH %d/%d" % (epoch+1, EPOCHS))
		

		# exponential learning rate decay
		# ogni 10 epoche come nel paper
		if (epoch + 1) % 10 == 0:
			sess.run(update_learning_rate, feed_dict={learning_rate_decay: 2.0})
		


	

		# initialize training dataset and set batch normalization training
		sess.run(train_initialization, feed_dict={data_features:x_train, data_labels:y_train, batch_size:BATCHSIZE})
		sess.run(metrics_initializer)
		sess.run(switch_training_inference)
		
		
		

		# Training the network
		
		print(" ")
		print("The dataset used for the training is "+str(DATASET)+" with a network of type "+str(NETWORK))
		print("It will be trained for "+str(EPOCHS)+ " epochs") 
		print("With initial Learning Rate = "+str(LR))
		print("Shift-Based ADAMAX optimizer is active now") if SWITCH else print("Vanilla Adam optimizer is active now")
		print(" ")

		for i in tqdm(range(NUM_BATCHES_TRAIN)):
			sess.run(train_op)	# train network on a single batch
			batch_trn_loss, _ = sess.run(metrics_update)
			training_loss, training_accuracy = sess.run(metrics)


		print("")
		print("Training loss: ",round(training_loss,3), " Training accuracy: ", round(training_accuracy,3))
		print("")
		
		summary = sess.run(merged_summary)
		train_writer.add_summary(summary, epoch)
		

		
		# Initialize the test dataset and set batc normalization inference
		sess.run(test_initialization, feed_dict={data_features:x_test, data_labels:y_test, batch_size:BATCHSIZE})
		sess.run(metrics_initializer)
		# switch between training phase and test phase
		sess.run(switch_training_inference)
		
		
		# Evaluation of the network
		for i in tqdm(range(NUM_BATCHES_TEST)):
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




	train_writer.close()
	test_writer.close()
	
	saver.save(sess, os.path.join(session_modeldir, 'model.ckpt'))

print('\nTraining completed!\nNetwork model is saved in  {}\nTraining logs are saved in {}'.format(session_modeldir, session_logdir))


