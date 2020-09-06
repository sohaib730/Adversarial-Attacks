
"""
    Author =  Sohaib kiani
    Projected Gradient Descent based attack. Effective both in white/black box setting

""" 


import tensorflow as tf
import numpy as np

class PGDAttack:
  def __init__(self, sess,model, batch_size):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.sess=sess
    self.epsilon = 0.05
    self.k = 5
    self.a = 0.01
    self.batch_shape= [batch_size, 32, 32, 3]
    self.batch_size=batch_size
    self.rand=True

    #logits = self.output

    self.x_input = tf.placeholder(tf.float32, shape=self.batch_shape)
    self.y_input = tf.placeholder(tf.int64, shape=[self.batch_size])
    self.output=self.model.predict(self.x_input)

    loss_func='xent'
    #self.output=self.model.predict(x)
    #logits = self.output

    if loss_func == 'xent':

      cost =  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.output)
      loss=tf.reduce_sum(cost)
    elif loss_func == 'cw':
      label_mask = tf.one_hot(y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * self.output, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * self.output, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, self.x_input)[0]

  def preturb(self, x_nat, y):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    sess=self.sess



    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
      x = np.copy(x_nat)

    for i in range(self.k):
      grad = sess.run(self.grad, feed_dict={self.x_input: x,self.y_input: y})

      x += self.a * np.sign(grad)

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, -0.5, 0.5) # ensure valid pixel range

    return x

  def attack(self,images,labels):
    adv_images=[]
    for batch in range(0,np.shape(images)[0],self.batch_size):
      adv_images_batch = self.preturb(images[batch:self.batch_size+batch,:,:,:],labels[batch:self.batch_size+batch])
      if batch is 0:
        adv_images=adv_images_batch
      else:
        print("here")
        adv_images=np.append(adv_images,adv_images_batch,axis=0)
    return adv_images
