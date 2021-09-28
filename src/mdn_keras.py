"""
A Mixture Density Layer for Keras
cpmpercussion: Charles Martin (University of Oslo) 2018
https://github.com/cpmpercussion/keras-mdn-layer
Hat tip to [Omimo's Keras MDN layer](https://github.com/omimo/Keras-MDN)
for a starting point for this code.
Provided under MIT License
"""

import tensorflow as tf
tf.random.set_seed(42)
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras import backend as K
def swish(x, beta = 1):
	return (x * K.sigmoid(beta * x))
def elu_plus(x):
	"""ELU activation with a very small addition to help prevent
	NaN in loss."""
	return K.elu(x) + 1 + K.epsilon()
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish': Activation(swish), 'elu_plus': Activation(elu_plus)})
from tensorflow_probability import distributions as tfd


class MDN_Keras(layers.Layer):
	"""A Mixture Density Network Layer for Keras.
	This layer has a few tricks to avoid NaNs in the loss function when training:
		- Activation for variances is ELU + 1 + 1e-8 (to avoid very small values)
		- Mixture weights (pi) are trained in as logits, not in the softmax space.
	A loss function needs to be constructed with the same output dimension and number of mixtures.
	A sampling function is also provided to sample from distribution parametrised by the MDN outputs.
	"""

	def __init__(self, output_dimension, num_mixtures, **kwargs):
		self.output_dim = output_dimension
		self.num_mix = num_mixtures
		with tf.name_scope('MDN_Keras'):
			self.mdn_mus = layers.Dense(self.num_mix * self.output_dim, name='mdn_mus')  # mix*output vals, no activation
			self.mdn_sigmas = layers.Dense(self.num_mix * self.output_dim, activation='elu_plus', name='mdn_sigmas')  # mix*output vals exp activation
			self.mdn_pi = layers.Dense(self.num_mix, name='mdn_pi')  # mix vals, logits
		super(MDN_Keras, self).__init__(**kwargs)

	def build(self, input_shape):
		with tf.name_scope('mus'):
			self.mdn_mus.build(input_shape)
		with tf.name_scope('sigmas'):
			self.mdn_sigmas.build(input_shape)
		with tf.name_scope('pis'):
			self.mdn_pi.build(input_shape)
		super(MDN_Keras, self).build(input_shape)

	@property
	def trainable_weights(self):
		return self.mdn_mus.trainable_weights + self.mdn_sigmas.trainable_weights + self.mdn_pi.trainable_weights

	@property
	def non_trainable_weights(self):
		return self.mdn_mus.non_trainable_weights + self.mdn_sigmas.non_trainable_weights + self.mdn_pi.non_trainable_weights

	def call(self, x, mask=None):
		with tf.name_scope('MDN_Keras'):
			mdn_out = layers.concatenate([self.mdn_mus(x),
										  self.mdn_sigmas(x),
										  self.mdn_pi(x)],
										 name='mdn_outputs')
		return mdn_out

	def compute_output_shape(self, input_shape):
		"""Returns output shape, showing the number of mixture parameters."""
		return (input_shape[0], (2 * self.output_dim * self.num_mix) + self.num_mix)

	def get_config(self):
		config = {
			"output_dimension": self.output_dim,
			"num_mixtures": self.num_mix
		}
		base_config = super(MDN_Keras, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

def get_mixture_loss_func(output_dim, num_mixes):
	"""Construct a loss functions for the MDN layer parametrised by number of mixtures."""
	# Construct a loss function with the right number of mixtures and outputs
	def mdn_loss_func(y_true, y_pred):
		# Reshape inputs in case this is used in a TimeDistribued layer
		y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
		y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
		# Split the inputs into paramaters
		out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
																		 num_mixes * output_dim,
																		 num_mixes],
											 axis=-1, name='mdn_coef_split')
		# Construct the mixture models
		cat = tfd.Categorical(logits=out_pi)
		component_splits = [output_dim] * num_mixes
		mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
		sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
		coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
				in zip(mus, sigs)]
		mixture = tfd.Mixture(cat=cat, components=coll)
		loss = mixture.log_prob(y_true)
		loss = tf.negative(loss)
		loss = tf.reduce_mean(loss)
		return loss

	# Actually return the loss function
	with tf.name_scope('MDN_Keras'):
		return mdn_loss_func


def get_mixture_sampling_fun(output_dim, num_mixes):
	"""Construct a TensorFlor sampling operation for the MDN layer parametrised
	by mixtures and output dimension. This can be used in a Keras model to
	generate samples directly."""

	def sampling_func(y_pred):
		# Reshape inputs in case this is used in a TimeDistribued layer
		y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
		out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
																		 num_mixes * output_dim,
																		 num_mixes],
											 axis=1, name='mdn_coef_split')
		cat = tfd.Categorical(logits=out_pi)
		component_splits = [output_dim] * num_mixes
		mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
		sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
		coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
				in zip(mus, sigs)]
		mixture = tfd.Mixture(cat=cat, components=coll)
		samp = mixture.sample()
		# Todo: temperature adjustment for sampling function.
		return samp

	# Actually return the loss_func
	with tf.name_scope('MDNLayer'):
		return sampling_func


def get_mixture_mse_accuracy(output_dim, num_mixes):
	"""Construct an MSE accuracy function for the MDN layer
	that takes one sample and compares to the true value."""
	# Construct a loss function with the right number of mixtures and outputs
	def mse_func(y_true, y_pred):
		# Reshape inputs in case this is used in a TimeDistribued layer
		y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
		y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
		out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
																		 num_mixes * output_dim,
																		 num_mixes],
											 axis=1, name='mdn_coef_split')
		cat = tfd.Categorical(logits=out_pi)
		component_splits = [output_dim] * num_mixes
		mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
		sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
		coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
				in zip(mus, sigs)]
		mixture = tfd.Mixture(cat=cat, components=coll)
		samp = mixture.sample()
		mse = tf.reduce_mean(tf.square(samp - y_true), axis=-1)
		# Todo: temperature adjustment for sampling functon.
		return mse

	# Actually return the loss_func
	with tf.name_scope('MDNLayer'):
		return mse_func


def split_mixture_params(params, output_dim, num_mixes):
	"""Splits up an array of mixture parameters into mus, sigmas, and pis
	depending on the number of mixtures and output dimension.
	Arguments:
	params -- the parameters of the mixture model
	output_dim -- the dimension of the normal models in the mixture model
	num_mixes -- the number of mixtures represented
	"""
	mus = params[:num_mixes * output_dim]
	sigs = params[num_mixes * output_dim:2 * num_mixes * output_dim]
	pi_logits = params[-num_mixes:]
	return mus, sigs, pi_logits


def softmax(w, t=1.0):
	"""Softmax function for a list or numpy array of logits. Also adjusts temperature.
	Arguments:
	w -- a list or numpy array of logits
	Keyword arguments:
	t -- the temperature for to adjust the distribution (default 1.0)
	"""
	e = np.array(w) / t  # adjust temperature
	e -= e.max()  # subtract max to protect from exploding exp values.
	e = np.exp(e)
	dist = e / np.sum(e)
	return dist


def sample_from_categorical(dist):
	"""Samples from a categorical model PDF.
	Arguments:
	dist -- the parameters of the categorical model
	Returns:
	One sample from the categorical model, or -1 if sampling fails.
	"""
	r = np.random.rand(1)  # uniform random number in [0,1]
	accumulate = 0
	for i in range(0, dist.size):
		accumulate += dist[i]
		if accumulate >= r:
			return i
	tf.logging.info('Error sampling categorical model.')
	return -1

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
	"""Calculates the negative log likelihood of the data given the inverse gamma parameters."""
	N = K.int_shape(target)[1]
	alpha, sigma = tf.split(output, num_or_size_splits=[N,N], axis=1)
	#K.print_tensor(alpha)
	#K.print_tensor(sigma)
	x = target / sigma
	neg_log_like = K.sum(K.log(sigma) + 1/x + tf.math.lgamma(alpha) + (alpha+1)*K.log(x))
	return K.mean(neg_log_like)