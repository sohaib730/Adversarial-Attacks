"""
Runs CleverHans attacks on Wide RESNET  or Madry Lab CIFAR-10 challenge model

1.Usage of cleverhans attacks
1. How to to model wrapper that enable cleverhans attacks
2. Generate adversarial examples for deepFool attacks in batches

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import logging

import tensorflow as tf
from tensorflow.python.platform import app, flags
from cleverhans.utils import set_log_level, to_categorical
from cleverhans.utils_tf import model_eval
import cifar10_input
from PIL import Image
import math
import pickle

FLAGS = flags.FLAGS


def main(argv):

  model_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

  if model_file is None:
    print('No model found')
    sys.exit()

  cifar = cifar10_input.CIFAR10Data(FLAGS.dataset_dir)

  nb_classes = 10
  X_test = cifar.train_data.xs
  Y_test = to_categorical(cifar.train_data.ys, nb_classes)
  assert Y_test.shape[1] == 10.
  print ("train data shape",X_test.shape)

  set_log_level(logging.DEBUG)

  with tf.Session() as sess:

    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    from cleverhans.model_zoo.madry_lab_challenges.cifar10_model import make_wresnet
    model = make_wresnet()
    saver = tf.train.Saver()
    # Restore the checkpoint
    saver.restore(sess, model_file)
    SCOPE = "cifar10_challenge"
    model2 = make_wresnet(scope=SCOPE)
    assert len(model.get_vars()) == len(model2.get_vars())
    found = [False] * len(model2.get_vars())
    for var1 in model.get_vars():
      var1_found = False
      var2_name = SCOPE + "/" + var1.name
      for idx, var2 in enumerate(model2.get_vars()):
        if var2.name == var2_name:
          var1_found = True
          found[idx] = True
          sess.run(tf.assign(var2, var1))
          break
      assert var1_found, var1.name
    assert all(found)

    model = model2
    saver = tf.train.Saver()

    # Restore the checkpoint
    #saver.restore(sess, model_file)

    nb_samples = FLAGS.nb_samples

    attack_params = {'batch_size': FLAGS.batch_size,
                     'clip_min': 0., 'clip_max': 255.}

    if FLAGS.attack_type == 'cwl2':
      from cleverhans.attacks import CarliniWagnerL2
      attacker = CarliniWagnerL2(model, sess=sess)
      attack_params.update({'binary_search_steps': 1,
                            'confidence':0,
                            'max_iterations': 100,
                            'learning_rate': 0.1,
                            'initial_const': 10,
                            'batch_size': 10
                            })

    else:  # eps and eps_iter in range 0-255
      attack_params.update({'eps': 16, 'ord': np.inf})
      if FLAGS.attack_type == 'fgsm':
        from cleverhans.attacks import FastGradientMethod
        attacker = FastGradientMethod(model, sess=sess)

      elif FLAGS.attack_type == 'pgd':
        attack_params.update({'eps':8,'eps_iter': .02,'ord':np.inf, 'nb_iter': 10})
        from cleverhans.attacks import MadryEtAl
        attacker = MadryEtAl(model, sess=sess)
      elif FLAGS.attack_type == 'deepFool':
        print ("here")
        attack_params.update({'ord':np.inf,'eps':6.0,'max_iter': 100})
        from CdeepFool_cleverhans import DeepFool

        attacker = DeepFool(model, sess=sess)

    eval_par = {'batch_size': FLAGS.batch_size}

    if FLAGS.sweep:
      max_eps = 16
      epsilons = np.linspace(1, max_eps, max_eps)
      for e in epsilons:
        t1 = time.time()
        attack_params.update({'eps': e})
        x_adv = attacker.generate(x, **attack_params)
        preds_adv = model.get_probs(x_adv)
        x1 = sess.run(x_adv,feed_dict = {x:X_test[0],y:Y_test[0]})
        print (x1.shape)
        l_inf = np.amax(np.abs(X_test[0] - x1))
        print ('perturbation found: {}'.format(l_inf))

        acc = model_eval(sess, x, y, preds_adv, X_test[
            :nb_samples], Y_test[:nb_samples], args=eval_par)
        print('Epsilon %.2f, accuracy on adversarial' % e,
              'examples %0.4f\n' % acc)
      t2 = time.time()
    else:
      t1 = time.time()
      x_adv = attacker.generate(x, **attack_params)
      preds_adv = model.get_probs(x_adv)
      logits  = model.get_logits(x)
      #print (len(x_adv))

      num_eval_examples = 1000
      eval_batch_size = 100
      num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

      x_adv_all = [] # adv accumulator
      y_adv_all = []
      y_true = []
      print('Iterating over {} batches'.format(num_batches))

      for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)
        print('batch size: {}'.format(bend - bstart))

        x_batch = X_test[bstart:bend, :]
        y_batch = Y_test[bstart:bend]
        y_clean = np.argmax(sess.run (logits,feed_dict={x:x_batch}),axis=1)
        x_b_adv,pred = sess.run([x_adv,preds_adv],feed_dict = {x:x_batch,y:y_batch})
        y_b_adv = np.argmax(sess.run (logits,feed_dict={x:x_b_adv}),axis=1)

        count = 0
        y_batch = np.argmax(y_batch,axis=1)
        for i in range(eval_batch_size):

            if (y_b_adv[i] != y_batch[i] and y_clean[i] == y_batch[i]):
                l_inf = np.amax(np.abs(x_batch[i] - x_b_adv[i]))
                print ('perturbation found: {}'.format(l_inf))
                #print (y_b_adv[i])
                x_adv_all.append(x_b_adv[i])
                y_adv_all.append(y_b_adv[i])
                y_true.append(y_batch[i])
                count +=1
        #print (y_adv_all[0:20])
        #print (y_true[0:20])
        print ("Totat adversariak cound in this batch",count)




        #x_adv_all.extend(x_b_adv)
        #y_adv_all.extend(y_b_adv)

      x_adv_all = np.array(x_adv_all)
      y_true = np.array(y_true)
      y_adv_all = np.array(y_adv_all)

      print ('Adv Label',y_adv_all[0:20])
      print ('Ori Label',y_true[0:20])

      #y_adv = np.squeeze(y_adv)
      print (x_adv_all.shape)
      print (y_adv_all.shape)
      print (y_true.shape)


      count = 0
      for i in range(y_adv_all.shape[0]):
          if y_true[i] != y_adv_all[i]:
              count+=1
      print ("Total adversarial examples found",count)
      pickle.dump((x_adv_all, y_true,y_adv_all), open('/scratch/kiani/Projects/CIFAR data/Adversarial/deepFool/iter_100/deepFool_E6_train.p', 'wb'))


      #from numpy import linalg as LA
      #l_2 = LA.norm(X_test[0] - x1[0])

      #l_inf = np.amax(np.abs(x - x_adv))







      t2 = time.time()
    print ("Range of data should be 0-255 and actual is: ",str(np.min(x_adv_all))+" "+str(np.max(x_adv_all)))
    image=((x_adv_all[2])).astype(np.uint8)
    img=Image.fromarray(image)
    img.save("deepFool_attack.jpeg")
    print("Took", t2 - t1, "seconds")


if __name__ == '__main__':


  cifar10_root = '/scratch/kiani/Projects/BlackBox_cifar10_challenge'
  default_ckpt_dir = os.path.join(cifar10_root, 'models/adv_trained_mine')
  default_data_dir = os.path.join(cifar10_root, 'cifar10_data')

  flags.DEFINE_integer('batch_size', 0, "Batch size")

  flags.DEFINE_integer('nb_samples',0 , "Number of samples to test")

  flags.DEFINE_string('attack_type', 'deepFool', ("Attack type: 'fgsm'->'fast "
                                              "gradient sign method', "
                                              "'pgd'->'projected "
                                              "gradient descent', 'cwl2'->"
                                              "'Carlini & Wagner L2'"))
  flags.DEFINE_string('checkpoint_dir', default_ckpt_dir,
                      'Checkpoint directory to load')

  flags.DEFINE_string('dataset_dir', default_data_dir, 'Dataset directory')

  flags.DEFINE_bool('sweep', False, 'Sweep epsilon or single epsilon?')

  app.run(main)
