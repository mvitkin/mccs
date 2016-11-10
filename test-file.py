from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import scipy.io
import os
import random

import tensorflow as tf

def main():
  directory = 'train_1'

  # Create the model
  x = tf.placeholder(tf.float32, [None, 3840000])
  W = tf.Variable(tf.zeros([3840000, 2]))
  b = tf.Variable(tf.zeros([2]))
  y = tf.matmul(x, W) + b

  # Placeholder for real classes
  y_ = tf.placeholder(tf.float32, [None, 2])

  # Loss (cross-entropy)
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()

  tf.initialize_all_variables().run()

  input_data = []
  target_data = []

  print('Retrieving data...')

  dirlist = os.listdir(directory)

  for i in range(500):
    input_data = []
    target_data = []
    batch = [ dirlist[p] for p in random.sample(xrange(len(dirlist)), 100) ]
    for filename in batch:
      path = os.path.join(directory/filename)
      print(path)
      mat_file = scipy.io.loadmat(path)

      label = int(filename[-5])
      data = mat_file['dataStruct']['data'][0][0]
      num_time_samples = mat_file['dataStruct']['nSamplesSegment'][0][0][0][0]
      sampling_rate = mat_file['dataStruct']['iEEGsamplingRate'][0][0][0][0]
      channel_indices = mat_file['dataStruct']['channelIndices'][0][0][0]
      sequence_index = mat_file['dataStruct']['sequence'][0][0][0][0]

      label_one_hot = [0, 0]
      label_one_hot[label] = 1;

      data = data.flatten()
      input_data.append(data)
      target_data.append(label_one_hot)


    print('Training on Batch ' + str(i))
    sess.run(train_step, feed_dict={x: input_data, y_: target_data})
    # print ("Epoch " + str(i) + " | " + str(W.eval()) + " | " + str(b.eval()))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: input_data,
                                        y_: target_data}))
    


if __name__ == '__main__':
  main()

'''
def main(_):
  data = scipy.io.loadmat('train_1/1_1_0.mat')
  
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  # Train
  tf.initialize_all_variables().run()
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()
'''