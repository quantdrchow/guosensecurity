# -*- coding: utf-8 -*-
"""
Created on      2019-6-19 17:29 

@author: zhouxianzheng
@Guosen Securities
"""







def model(x, training, scope='model'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu,
          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.04))
    x = tf.layers.max_pooling2d(x, (2, 2), 1)
    x = tf.layers.flatten(x)
    x = tf.layers.dropout(x, 0.1, training=training)
    x = tf.layers.dense(x, 64, activation=tf.nn.relu)
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.layers.dense(x, 10, activation=tf.nn.softmax)
    return x

train_out = model(train_data, training=True)
test_out = model(test_data, training=False)



def model(x, training, scope='model'):
  with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
    x = tf.compat.v1.layers.conv2d(x, 32, 3, activation=tf.nn.relu,
          kernel_regularizer=tf.keras.regularizers.l2(0.5 * (0.04)))
    x = tf.compat.v1.layers.max_pooling2d(x, (2, 2), 1)
    x = tf.compat.v1.layers.flatten(x)
    x = tf.compat.v1.layers.dropout(x, 0.1, training=training)
    x = tf.compat.v1.layers.dense(x, 64, activation=tf.nn.relu)
    x = tf.compat.v1.layers.batch_normalization(x, training=training)
    x = tf.compat.v1.layers.dense(x, 10, activation=tf.nn.softmax)
    return x

train_out = model(train_data, training=True)
test_out = model(test_data, training=False)





