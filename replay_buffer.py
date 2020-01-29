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

	def __init__(self, action_size, buffer_size, batch_size, seed, alpha=1.0, beta=None, eps=0.01):
		"""Initialize a ReplayBuffer object.

		Params
		======
			action_size (int): dimension of each action
			buffer_size (int): maximum size of buffer
			batch_size (int): size of each training batch
			seed (int): random seed
			alpha (float): interpolation factor for sampling; 0 -> random uniform sampling, 1 -> priority sampling
			beta (float): factor for update rule; 0 -> no modified update, 1 -> full correction for bias introduced
			by prio sampling
		"""
		self.action_size = action_size
		self.buffer_size = buffer_size
		self.memory = deque(maxlen=buffer_size)
		self.priority_memory = NumpyStorage(buffer_size)
		self.batch_size = batch_size
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
		self.seed = random.seed(seed)
		self.alpha = alpha
		self.beta = beta
		self.eps = eps

	def add(self, state, action, reward, next_state, done, priority=1.0):
		"""Add a new experience to memory."""
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)
		self.priority_memory.append(priority)

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

		return (states, actions, rewards, next_states, dones, experiences_indices)

	def update_probability(self, delta, indices):
		delta = abs(delta) + self.eps
		self.priority_memory[indices] = delta

	def __len__(self):
		"""Return the current size of internal memory."""
		return len(self.memory)

	def compute_probs(self, indices=None):
		if indices is None:
			priorities = self.priority_memory[:]
		else:
			priorities = self.priority_memory[indices]
		if len(priorities) == 0:
			return np.array([])

		probabilites = np.array([prio**self.alpha for prio in priorities]).reshape(-1)

		return 1/np.sum(probabilites) * probabilites

	def compute_weights(self, experiences_indices):
		probabilities = self.compute_probs(indices=experiences_indices)
		beta = next(self.beta)
		weights = (probabilities * len(self.memory)) ** (-beta)
		return 1.0 / np.max(weights) * weights


class NumpyStorage:
	def __init__(self, maxlen):
		self.maxlen = maxlen
		self.storage = np.array([np.nan] * maxlen)
		self.index = 0

	def append(self, element):
		if self.index == self.maxlen:
			self.storage[:-1] = self.storage[1:]
			self.index -= 1

		self.storage[self.index] = element
		self.index += 1
		return self

	def __getitem__(self, key):
		return self.storage[:self.index][key]

	def __setitem__(self, key, value):
		self.storage[key] = value
		return self
