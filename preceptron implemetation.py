import tensorflow as tf
import numpy as np
print('done importing packages')


n_inputs= 28 * 28
n_hidden1= 300
n_hidden2= 100
n_outputs= 10

X= tf.placeholder(tf.float32, shape= (None,n_inputs), name= 'X')
y= tf.placeholder(tf.int32, shape=None, name= 'y')

with tf.name_scope('dnn'):
    hidden1= tf.layers.dense(X, n_hidden1, name= 'n_hidden1',
                             activation= tf.nn.relu)
    hidden2= tf.layers.dense(hidden1, n_hidden2, name= 'n_hidden2',
                             activation= tf.nn.relu)
    logits= tf.layers.dense(hidden2, n_outputs, name= 'n_outputs')

with tf.name_scope('loss'):
    xentropy= tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y
                                                             ,logits= logits)
    loss= tf.reduce_mean(xentropy, name='loss')

learning_rate= 0.01
with tf.name_scope('train'):
   optimizer= tf.train.GradientDescentOptimizer(learning_rate= learning_rate)
   training_op= optimizer.minimize(loss)


with tf.name_scope('eval'):
    correct= tf.nn.in_top_k(logits, y, 1)
    accuracy= tf.reduce_mean(tf.cast(correct, tf.float32))

init= tf.global_variables_initializer()
saver= tf.train.Saver()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

n_epochs= 40
batch_size= 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for it in range(mnist.train.num_examples // batch_size):
            x_batch, y_batch= mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X:x_batch, y:y_batch})
        acc_train= accuracy.eval(feed_dict= {X:x_batch, y:y_batch})
        acc_val= accuracy.eval(feed_dict= {X:mnist.validation.images,
                                           y:mnist.validation.images})
       # print(epoch, 'Training accuracy:', acc_train, 'val accuracy:', acc_val)
    #save_path= saver.save(sess, '/my model_final.cpkt')    
    
    
##                            
