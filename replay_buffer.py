import torch
import random
import numpy as np
from collections import namedtuple, deque


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
	"""Fixed-size buffer to store experience tuples."""

	def __init__(self, action_size, buffer_size, batch_size, seed):
		"""Initialize a ReplayBuffer object.

		Params
		======
			action_size (int): dimension of each action
			buffer_size (int): maximum size of buffer
			batch_size (int): size of each training batch
			seed (int): random seed
		"""
		self.action_size = action_size
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
		self.seed = random.seed(seed)

	def add(self, state, action, reward, next_state, done):
		"""Add a new experience to memory."""
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)

	def sample(self):
		"""Randomly sample a batch of experiences from memory."""
		experiences = random.choices(self.memory, k=self.batch_size)

		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
			device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
			device)

		return (states, actions, rewards, next_states, dones)

	def __len__(self):
		"""Return the current size of internal memory."""
		return len(self.memory)

class ReplayBufferWeighted:
	"""Fixed-size buffer to store experience tuples."""

	def __init__(self, action_size, buffer_size, batch_size, seed, alpha=1.0, beta=0.5, eps=0.01):
		"""Initialize a ReplayBuffer object.

		Params
		======
			action_size (int): dimension of each action
			buffer_size (int): maximum size of buffer
			batch_size (int): size of each training batch
			seed (int): random seed
		"""
		self.action_size = action_size
		self.buffer_size = buffer_size
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
		self.seed = random.seed(seed)
		self.alpha = alpha
		self.beta = beta
		self.eps = eps

	def add(self, state, action, reward, next_state, done, weight=1.0):
		"""Add a new experience to memory."""
		#print("adding", weight)
		e = self.experience(state, action, reward, next_state, done, weight)
		self.memory.append(e)

	def sample(self):
		"""Randomly sample a batch of experiences from memory."""
		experiences_indices = random.choices(range(len(self.memory)), weights=self.compute_probs(), k=self.batch_size)
		experiences = [self.memory[i] for i in experiences_indices]

		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
			device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
			device)
		priority = (np.array([e.priority for e in experiences if e is not None]).astype(np.uint8))

		return (states, actions, rewards, next_states, dones, priority, experiences_indices)

	def update_probability(self, delta, indices):
		#print("delta", delta)
		#print("indices", indices)
		delta = abs(delta) + self.eps
		for i_c, i in enumerate(indices):
			#print("weight", delta[i_c])
			self.memory[i] = self.memory[i]._replace(priority=delta[i_c])

	def __len__(self):
		"""Return the current size of internal memory."""
		return len(self.memory)

	def _priority(self):
		return [e.priority for e in self.memory]

	def compute_probs(self, indices=None):
		if indices is None:
			priorities = np.array([priority**self.alpha for priority in self._priority()]).reshape(-1)
		else:
			priorities = self._priority()
			priorities = np.array([priorities[i]**self.alpha for i in indices]).reshape(-1)
		#print("weights in compute probs", weights)
		#print("sum", 1/np.sum(weights))
		if len(priorities) == 0:
			return np.array([])
		return 1/np.sum(priorities) * priorities

	def compute_weights(self, experiences_indices):
		probs = self.compute_probs(indices=experiences_indices)
		# print("probstype", type(probs))
		# print("probs", probs.shape)
		# print("buffer", self.memory.buffer_size)
		# print("alpha", self.memory.alpha)
		# print("weightstype", type(weights))
		# print("weights", weights.shape)
		weights = (probs * len(self.memory)) ** (-self.beta)  # * weights
		return 1.0 / np.max(weights) * weights  # * weights

