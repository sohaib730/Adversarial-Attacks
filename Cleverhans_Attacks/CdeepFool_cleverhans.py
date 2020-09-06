"""
The DeepFool attack

* This is CleverHans RobustML library code
* Importing this file for Small modifications
"""
import copy
import logging
import warnings

import numpy as np
import tensorflow as tf

from cleverhans.attacks import Attack
from cleverhans.model import Model, wrapper_warning_logits, CallableModelWrapper
from cleverhans import utils
from cleverhans import utils_tf
from cleverhans.utils_tf import clip_eta
from six.moves import xrange
np_dtype = np.dtype('float32')

_logger = utils.create_logger("cleverhans.attacks.deep_fool")
_logger.setLevel(logging.INFO)


def jacobian_graph(predictions, x, nb_classes):
  """
  Create the Jacobian graph to be ran later in a TF session
  :param predictions: the model's symbolic output (linear output,
      pre-softmax)
  :param x: the input placeholder
  :param nb_classes: the number of classes the model has
  :return:
  """

  # This function will return a list of TF gradients
  list_derivatives = []

  # Define the TF graph elements to compute our derivatives for each class
  for class_ind in xrange(nb_classes):
    derivatives, = tf.gradients(predictions[:, class_ind], x)
    list_derivatives.append(derivatives)

  return list_derivatives
class DeepFool(Attack):
  """
  DeepFool is an untargeted & iterative attack which is based on an
  iterative linearization of the classifier. The implementation here
  is w.r.t. the L2 norm.
  Paper link: "https://arxiv.org/pdf/1511.04599.pdf"
  :param model: cleverhans.model.Model
  :param sess: tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, sess, dtypestr='float32', **kwargs):
    """
    Create a DeepFool instance.
    """
    if not isinstance(model, Model):
      wrapper_warning_logits()
      model = CallableModelWrapper(model, 'logits')

    super(DeepFool, self).__init__(model, sess, dtypestr, **kwargs)

    self.structural_kwargs = [
        'eps','overshoot', 'max_iter', 'clip_max', 'clip_min', 'nb_candidate'
    ]

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.
    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    assert self.sess is not None, \
        'Cannot use `generate` when no `sess` was provided'
    #from cleverhans.utils_tf import jacobian_graph

    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    # Define graph wrt to this input placeholder
    logits = self.model.get_logits(x)
    self.nb_classes = logits.get_shape().as_list()[-1]
    assert self.nb_candidate <= self.nb_classes, \
        'nb_candidate should not be greater than nb_classes'
    preds = tf.reshape(
        tf.nn.top_k(logits, k=self.nb_candidate)[0],
        [-1, self.nb_candidate])
    # grads will be the shape [batch_size, nb_candidate, image_size]
    grads = tf.stack(jacobian_graph(preds, x, self.nb_candidate), axis=1)

    # Define graph
    def deepfool_wrap(x_val):
      """deepfool function for py_func"""
      return deepfool_batch(self.sess, x, preds, logits, grads, x_val,
                            self.eps,self.nb_candidate, self.overshoot,
                            self.max_iter, self.clip_min, self.clip_max,
                            self.nb_classes)

    wrap = tf.py_func(deepfool_wrap, [x], self.tf_dtype)
    wrap.set_shape(x.get_shape())
    return wrap

  def parse_params(self,
                   eps=8.0,
                   nb_candidate=10,
                   overshoot=0.02,
                   max_iter=50,
                   clip_min=0.,
                   clip_max=255.,
                   **kwargs):
    """
    :param nb_candidate: The number of classes to test against, i.e.,
                         deepfool only consider nb_candidate classes when
                         attacking(thus accelerate speed). The nb_candidate
                         classes are chosen according to the prediction
                         confidence during implementation.
    :param overshoot: A termination criterion to prevent vanishing updates
    :param max_iter: Maximum number of iteration for deepfool
    :param clip_min: Minimum component value for clipping
    :param clip_max: Maximum component value for clipping
    """
    self.eps = eps
    self.nb_candidate = nb_candidate
    self.overshoot = overshoot
    self.max_iter = max_iter
    self.clip_min = clip_min
    self.clip_max = clip_max
    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True


def deepfool_batch(sess,
                   x,
                   pred,
                   logits,
                   grads,
                   X,
                   eps,
                   nb_candidate,
                   overshoot,
                   max_iter,
                   clip_min,
                   clip_max,
                   nb_classes,
                   feed=None):
  """
  Applies DeepFool to a batch of inputs
  :param sess: TF session
  :param x: The input placeholder
  :param pred: The model's sorted symbolic output of logits, only the top
               nb_candidate classes are contained
  :param logits: The model's unnormalized output tensor (the input to
                 the softmax layer)
  :param grads: Symbolic gradients of the top nb_candidate classes, procuded
                from gradient_graph
  :param X: Numpy array with sample inputs
  :param nb_candidate: The number of classes to test against, i.e.,
                       deepfool only consider nb_candidate classes when
                       attacking(thus accelerate speed). The nb_candidate
                       classes are chosen according to the prediction
                       confidence during implementation.
  :param overshoot: A termination criterion to prevent vanishing updates
  :param max_iter: Maximum number of iteration for DeepFool
  :param clip_min: Minimum value for components of the example returned
  :param clip_max: Maximum value for components of the example returned
  :param nb_classes: Number of model output classes
  :return: Adversarial examples
  """
  X_adv = deepfool_attack(
      sess,
      x,
      pred,
      logits,
      grads,
      X,
      eps,
      nb_candidate,
      overshoot,
      max_iter,
      clip_min,
      clip_max,
      feed=feed)

  return np.asarray(X_adv, dtype=np_dtype)


def deepfool_attack(sess,
                    x,
                    predictions,
                    logits,
                    grads,
                    sample,
                    eps,
                    nb_candidate,
                    overshoot,
                    max_iter,
                    clip_min,
                    clip_max,
                    feed=None):
  """
  TensorFlow implementation of DeepFool.
  Paper link: see https://arxiv.org/pdf/1511.04599.pdf
  :param sess: TF session
  :param x: The input placeholder
  :param predictions: The model's sorted symbolic output of logits, only the
                     top nb_candidate classes are contained
  :param logits: The model's unnormalized output tensor (the input to
                 the softmax layer)
  :param grads: Symbolic gradients of the top nb_candidate classes, procuded
               from gradient_graph
  :param sample: Numpy array with sample input
  :param nb_candidate: The number of classes to test against, i.e.,
                       deepfool only consider nb_candidate classes when
                       attacking(thus accelerate speed). The nb_candidate
                       classes are chosen according to the prediction
                       confidence during implementation.
  :param overshoot: A termination criterion to prevent vanishing updates
  :param max_iter: Maximum number of iteration for DeepFool
  :param clip_min: Minimum value for components of the example returned
  :param clip_max: Maximum value for components of the example returned
  :return: Adversarial examples
  """
  adv_x = copy.copy(sample)
  # Initialize the loop variables
  iteration = 0
  current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)
  if current.shape == ():
    current = np.array([current])
  w = np.squeeze(np.zeros(sample.shape[1:]))  # same shape as original image
  r_tot = np.zeros(sample.shape)
  original = current  # use original label as the reference

  _logger.debug(
      "Starting DeepFool attack up to %s iterations", max_iter)
  # Repeat this main loop until we have achieved misclassification
  while (np.any(current == original) and iteration < max_iter):

    if iteration % 10 == 0 and iteration > 0:
      _logger.info("Attack result at iteration %s is %s", iteration, current)
    gradients = sess.run(grads, feed_dict={x: adv_x})
    predictions_val = sess.run(predictions, feed_dict={x: adv_x})
    for idx in range(sample.shape[0]):
      pert = np.inf
      if current[idx] != original[idx]:
        continue
      for k in range(1, nb_candidate):
        w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
        f_k = predictions_val[idx, k] - predictions_val[idx, 0]
        # adding value 0.00001 to prevent f_k = 0
        pert_k = (abs(f_k) + 0.00001) / np.linalg.norm(w_k.flatten())
        if pert_k < pert:
          pert = pert_k
          w = w_k
      r_i = pert * w / np.linalg.norm(w)
      r_tot[idx, ...] = r_tot[idx, ...] + r_i
    #eta = clip_eta(r_tot, 2, 8.0)
    adv_x = np.clip(r_tot + sample, clip_min, clip_max)
    adv_x = np.clip(adv_x, sample - eps, sample + eps)
    current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)
    if current.shape == ():
      current = np.array([current])
    # Update loop variables
    iteration = iteration + 1


  # need to clip this image into the given range
  pert_image = sample + (1+overshoot)*r_tot
  pert_image = np.clip(pert_image, sample - eps, sample + eps)
  adv_x = np.clip(pert_image, clip_min, clip_max)
  current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)
  if current.shape == ():
      current = np.array([current])

  # need more revision, including info like how many succeed
  _logger.info("Attack result at iteration %s is %s and original", iteration, current)
  _logger.info("%s out of %s become adversarial examples at iteration %s",
               sum(current != original),
               sample.shape[0],
               iteration)
  lab =  np.where(current != original)

  #print (lab)
  return adv_x
