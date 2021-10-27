import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras import backend as K
def swish(x, beta = 1):
	return (x * K.sigmoid(beta * x))
def elu_plus(x):
	return K.elu(x) + 1 + K.epsilon()
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish), 'elu_plus': Activation(elu_plus)})
from tensorflow_probability import distributions as tfd

'''
Keras Layer for fitting an inverse Gamma distribution. Inspried by:
cpmpercussion: Charles Martin (University of Oslo) 2018
https://github.com/cpmpercussion/keras-mdn-layer
'''


class InvGamma_Keras(layers.Layer):

	def __init__(self, **kwargs):
		with tf.name_scope('InvGamma_Keras'):
			self.alpha = layers.Dense(1, activation='elu_plus')
			self.sigma = layers.Dense(1, activation='elu_plus')
		super(InvGamma_Keras, self).__init__(**kwargs)

	def build(self, input_shape):
		with tf.name_scope('alpha'):
			self.alpha.build(input_shape)
		with tf.name_scope('sigma'):
			self.sigma.build(input_shape)

		super(InvGamma_Keras, self).build(input_shape)

	@property
	def trainable_weights(self):
		return self.alpha.trainable_weights + self.sigma.trainable_weights

	@property
	def non_trainable_weights(self):
		return self.alpha.non_trainable_weights + self.sigma.non_trainable_weights

	def call(self, x):
		with tf.name_scope('InvGamma_Keras'):
			out = layers.concatenate([self.alpha(x), self.sigma(x)])
		return out

def inv_gamma_loss(target, output):
	'''
		Calculates the negative log likelihood of the data given the inverse gamma parameters.
	'''
	N = K.int_shape(target)[1]
	alpha, sigma = tf.split(output, num_or_size_splits=[N,N], axis=1)
	#K.print_tensor(sigma)
	x = target / sigma
	neg_log_like = K.sum(K.log(sigma) + 1/x + tf.math.lgamma(alpha) + (alpha+1)*K.log(x))
	return K.mean(neg_log_like)