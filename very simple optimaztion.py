import tensorflow as tf
import numpy as np
print('done importing packages')

x= tf.placeholder(tf.float32, shape=[None, 3])
y_true= tf.placeholder(tf.float32, shape= None)
w= tf.Variable([[0,0,0]], dtype= tf.float32, name= 'weights')
b= tf.Variable(0 ,dtype= tf.float32, name= 'bias')

y_pred= tf.matmul(w, tf.transpose(x)) + b #remember the formular for linear regression

#defining a loss function
loss= tf.nn.sigmoid_cross_entropy_with_logits(labels= y_true, logits= y_pred)
loss = tf.reduce_mean(loss)

#optimizing
learning_rate= 0.5
optimizer= tf.train.GradientDescentOptimizer(learning_rate)
train= optimizer.minimize(loss)