"""A module for a mixture density network layer
For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math


ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)
LOG2PI = math.log(2 * math.pi)
EPS = 1e-15

class MDN(nn.Module):
	"""A mixture density network layer
	The input maps to the parameters of a MoG probability distribution, where
	each Gaussian has O dimensions and diagonal covariance.
	Arguments:
		in_features (int): the number of dimensions in the input
		out_features (int): the number of dimensions in the output
		num_gaussians (int): the number of Gaussians per output dimensions
	Input:
		minibatch (BxD): B is the batch size and D is the number of input
			dimensions.
	Output:
		(pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
			number of Gaussians, and O is the number of dimensions for each
			Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
			is the standard deviation of each Gaussian. Mu is the mean of each
			Gaussian.
	"""

	def __init__(self, in_features, out_features, num_gaussians):
		super(MDN, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.num_gaussians = num_gaussians
		self.pi = nn.Sequential(
			nn.Linear(in_features, num_gaussians),
			nn.Softmax(dim=1)
		)
		self.sigma = nn.Linear(in_features, out_features * num_gaussians)
		self.mu = nn.Linear(in_features, out_features * num_gaussians)

	def forward(self, x):
		pi = self.pi(x)
		sigma = nn.ELU()(self.sigma(x)) + 1 + EPS
		sigma = sigma.view(-1, self.num_gaussians, self.out_features)
		mu = self.mu(x)
		mu = mu.view(-1, self.num_gaussians, self.out_features)
		return pi, sigma, mu


def gaussian_probability(sigma, mu, target, log=False):
	"""Returns the probability of `target` given MoG parameters `sigma` and `mu`.
	Arguments:
		sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
			size, G is the number of Gaussians, and O is the number of
			dimensions per Gaussian.
		mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
			number of Gaussians, and O is the number of dimensions per Gaussian.
		target (BxI): A batch of target. B is the batch size and I is the number of
			input dimensions.
	Returns:
		probabilities (BxG): The probability of each point in the probability
			of the distribution in the corresponding sigma/mu index.
	"""
	target = target.expand_as(sigma)
	if log:
		ret = (-torch.log(sigma) - 0.5 * LOG2PI	- 0.5 * torch.pow((target - mu) / sigma, 2))
	else:
		ret = (ONEOVERSQRT2PI / sigma) * torch.exp(-0.5 * ((target - mu) / sigma) ** 2)
	return ret

def mdn_loss(pi, sigma, mu, target):
	"""Calculates the error, given the MoG parameters and the target
	The loss is the negative log likelihood of the data given the MoG
	parameters.
	"""
	log_component_prob = self.gaussian_probability(sigma, mu, target, log=True)
	log_mix_prob = torch.log(
		nn.functional.gumbel_softmax(pi, tau=1, dim=-1) + EPS
	)
	return -torch.logsumexp(log_component_prob + log_mix_prob, dim=-1)

def sample(self, pi, sigma, mu):
	"""Draw samples from a MoG."""
	categorical = Categorical(pi)
	pis = categorical.sample().unsqueeze(1)
	sample = Variable(sigma.data.new(sigma.size(0), 1).normal_())
	# Gathering from the n Gaussian Distribution based on sampled indices
	sample = sample * sigma.gather(1, pis) + mu.gather(1, pis)
	return sample

def generate_samples(self, pi, sigma, mu, n_samples=None):
	if n_samples is None:
		n_samples = self.hparams.n_samples
	samples = []
	softmax_pi = nn.functional.gumbel_softmax(pi, tau=1, dim=-1)
	assert (
		softmax_pi < 0
	).sum().item() == 0, "pi parameter should not have negative"
	for _ in range(n_samples):
		samples.append(self.sample(softmax_pi, sigma, mu))
	samples = torch.cat(samples, dim=1)
	return samples

def generate_point_predictions(self, pi, sigma, mu, n_samples=None):
	# Sample using n_samples and take average
	samples = self.generate_samples(pi, sigma, mu, n_samples)
	if self.hparams.central_tendency == "mean":
		y_hat = torch.mean(samples, dim=-1)
	elif self.hparams.central_tendency == "median":
		y_hat = torch.median(samples, dim=-1).values
	return y_hat


def sample(pi, sigma, mu):
	"""Draw samples from a MoG.
	"""
	# Choose which gaussian we'll sample from
	pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
	# Choose a random sample, one randn for batch X output dims
	# Do a (output dims)X(batch size) tensor here, so the broadcast works in
	# the next step, but we have to transpose back.
	gaussian_noise = torch.randn((sigma.size(2), sigma.size(0)), requires_grad=False)
	variance_samples = sigma.gather(1, pis).detach().squeeze()
	mean_samples = mu.detach().gather(1, pis).squeeze()
	return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)


class InvGamma(nn.Module):
	"""A density network layer
	The input maps to the parameters of an inverse gamma probability distribution.
	Arguments:
		in_features (int): the number of dimensions in the input
	Input:
		minibatch (B): B is the batch size.
	Output:
		(alpha, sigma, mu) (B, B, B): B is the batch size. Alpha is the parameter of Gamma. 
		Sigma is the scale of the Gamma. Mu is the shift of the Gamma.
	"""

	def __init__(self, in_features):
		super(InvGamma, self).__init__()
		self.in_features = in_features
		self.alpha = nn.Linear(in_features, 1)
		self.sigma = nn.Linear(in_features, 1)
		#self.mu = nn.Linear(in_features, 1)
		self.elu = nn.ELU()

	def forward(self, x):
		alpha = self.elu(self.alpha(x)) + 1 + EPS
		sigma = self.elu(self.sigma(x)) + 1 + EPS
		#mu = self.elu(self.mu(x)) + 1 + EPS
		return alpha, sigma

def inv_gamma_loss(alpha, sigma, target):
	"""Calculates the negative log likelihood of the data given the inverse gamma parameters."""
	x = target / sigma
	neg_log_like = torch.sum(torch.log(sigma) + 1/x + torch.lgamma(alpha) + (alpha+1)*torch.log(x))
	return torch.mean(neg_log_like)