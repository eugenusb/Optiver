import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math

# Pytorch Layer to fit an inverse Gamma distribution. Inspired by https://github.com/sagelywizard/pytorch-mdn

EPS = 1e-15

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