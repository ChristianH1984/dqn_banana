import torch
from torch import optim
import torch.nn as nn
from collections import OrderedDict
import copy


class QNetwork(nn.Module):
	"""Actor (Policy) Model."""

	def __init__(self, state_size, action_size, lr=1e-5, seed=1337):
		"""Initialize parameters and build model.
		Params
		======
			state_size (int): Dimension of each state
			action_size (int): Dimension of each action
			seed (int): Random seed
		"""
		super(QNetwork, self).__init__()
		self.seed = torch.manual_seed(seed)
		"*** YOUR CODE HERE ***"
		input_size = state_size
		hidden_sizes = [1024, 512]
		output_size = action_size

		# Build a feed-forward network
		self.model = nn.Sequential(OrderedDict([
			('fc1', nn.Linear(input_size, hidden_sizes[0])),
			('relu1', nn.ReLU()),
			('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
			('relu2', nn.ReLU()),
			('logits', nn.Linear(hidden_sizes[1], output_size))
		]))

		self.loss = nn.MSELoss()
		self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

	def forward(self, state):
		"""Build a network that maps state -> action values."""
		return self.model.forward(state)

	def optimize(self, y_orig, y_target, weights=None):
		if weights is not None:
			#print("w in loss", weights.shape)
			#print("loss", self.loss(y_orig, y_target).shape)
			loss = weighted_mse_loss(y_orig, y_target, weights)#self.loss(y_orig, y_target).squeeze() * weights
		else:
			loss = self.loss(y_orig, y_target)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def clone(self):
		state_size = self.model.fc1.weight.shape[1]
		action_size = self.model.logits.weight.shape[0]
		qnetwork_clone = QNetwork(state_size, action_size)
		qnetwork_clone.model = copy.deepcopy(self.model)
		return qnetwork_clone

def weighted_mse_loss(input, target, weights):
	out = (input - target) ** 2
	out = out * weights.expand_as(out)
	loss = out.mean(0)
	return loss
