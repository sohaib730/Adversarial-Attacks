

"""
Author ==  Sohaib kiani
Implementation of highly effective Black Box attack

"""

import tensorflow as tf
import numpy as np


class BlackBoxMomentum:
  def __init__(self, sess, model, batch_size=1, max_iterations = 10,max_epsilon=16):
    # Images for CIFAR classifier are normalized to be in [0, 1] interval,
    # eps is a difference between pixels so it should be in [0, 1] interval.
    # Renormalizing epsilon from [0, 255] to [0, 1].
    self.eps = max_epsilon / 255.0
    self.batch_shape = [batch_size, 32, 32, 3]
    self.batch_size=batch_size
    self.model=model
    self.sess=sess
    self.num_iter=max_iterations


  def graph(self,x, y, i, x_max, x_min, grad):
    eps = self.eps
    num_iter = self.num_iter
    alpha = eps / num_iter
    momentum = 1.0
    num_classes = 10


    # prediction BEFORE-SOFTMAX of the model
    self.output = self.model.predict(x)

    pred = tf.argmax( self.output, 1)
    first_round = tf.cast(tf.equal(i, 0), tf.int64)
    y = first_round * pred + (1 - first_round) * y
    one_hot = tf.one_hot(y, num_classes)


    logits = self.output

    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits,label_smoothing=0.0, weights=0.4)

    noise = tf.gradients(cross_entropy, x)[0]
    noise = noise / tf.reduce_mean(tf.abs(noise), [1,2,3], keepdims=True)
    grad = momentum * grad + noise
    x = x + alpha * tf.sign(grad)
    x = tf.clip_by_value(x, x_min, x_max)
    #noise=tf.print(noise,[noise],message="Noise")
    i = tf.add(i, 1)
    return x, y, i, x_max, x_min, noise

  def stop(self,x, y, i, x_max, x_min, grad):
    num_iter = self.num_iter

    return tf.less(i, num_iter)



  def attack(self,images,labels):
    x_input = tf.placeholder(tf.float32, shape=self.batch_shape)
    x_max = tf.clip_by_value(x_input + self.eps, -0.5, 0.5)
    x_min = tf.clip_by_value(x_input - self.eps, -0.5, 0.5)

    y = tf.constant(np.zeros([self.batch_size]), tf.int64)
    i = tf.constant(0)
    grad = tf.zeros(shape=self.batch_shape)

    temp_images=images

    x_adv, _, _, _, _, _ = tf.while_loop(self.stop, self.graph, [x_input, y, i, x_max, x_min, grad])

    adv_images=[]
    for batch in range(0,np.shape(images)[0],self.batch_size):
      adv_images_batch = self.sess.run(x_adv, feed_dict={x_input: temp_images[batch:self.batch_size+batch,:,:,:]})
      if batch is 0:
        adv_images=adv_images_batch
      else:
        print("here")
        adv_images=np.append(adv_images,adv_images_batch,axis=0)
    return adv_images
