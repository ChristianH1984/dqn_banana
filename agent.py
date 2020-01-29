import numpy as np
import random
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseAgent:
	"""Interacts with and learns from the environment."""

	def __init__(self, gamma, tau, batch_size, update_every, qnetwork, memory):
		"""Initialize an Agent object.

		Params
		======
			gamma (float): discount factor of future rewards
			tau (float): soft update parameter
			batch_size (int): number of samples for each mini batch
			update_every (int): number of steps until weights are copied  from local to target Q-network
			state_size (int): dimension of each state
			action_size (int): dimension of each action
			seed (int): random seed
		"""
		# Q-Network
		self.gamma = gamma
		self.tau = tau
		self.batch_size = batch_size
		self.update_every = update_every
		self.qnetwork_local = qnetwork.to(device)
		self.qnetwork_target = qnetwork.clone().to(device)

		# Replay memory
		self.memory = memory

		# Initialize time step (for updating every UPDATE_EVERY steps)
		self.t_step = 0

	def step(self, state, action, reward, next_state, done):
		# Save experience in replay memory
		self.memory.add(state, action, reward, next_state, done)

		# Learn every UPDATE_EVERY time steps.
		self.t_step = (self.t_step + 1) % self.update_every
		if self.t_step == 0:
			# If enough samples are available in memory, get random subset and learn
			if len(self.memory) > self.batch_size:
				experiences = self.memory.sample()
				self.learn(experiences)

	def act(self, state, action_size, eps=0.):
		"""Returns actions for given state as per current policy.

		Params
		======
			state (array_like): current state
			eps (float): epsilon, for epsilon-greedy action selection
		"""
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.qnetwork_local.eval()
		with torch.no_grad():
			action_values = self.qnetwork_local(state)
		self.qnetwork_local.train()

		# Epsilon-greedy action selection
		if random.random() > eps:
			return np.argmax(action_values.cpu().data.numpy()).astype(int)
		else:
			return random.choice(np.arange(action_size)).astype(int)

	def learn(self, experiences):
		"""Update value parameters using given batch of experience tuples.

		Params
		======
			experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
		"""
		pass

	def soft_update(self, local_model, target_model):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target

		Params
		======
			local_model (PyTorch model): weights will be copied from
			target_model (PyTorch model): weights will be copied to
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

class DQNAgent(BaseAgent):
	"""Interacts with and learns from the environment."""

	def learn(self, experiences):
		"""Update value parameters using given batch of experience tuples.

		Params
		======
			experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
		"""
		states, actions, rewards, next_states, dones = experiences
		y_target = rewards + self.gamma * self.qnetwork_target(next_states).max(dim=1, keepdim=True)[0] * (1 - dones)
		self.qnetwork_local.optimize(self.qnetwork_local(states).gather(1, actions), y_target.detach())

		self.soft_update(self.qnetwork_local, self.qnetwork_target)


class DoubleDQNAgent(BaseAgent):
	"""Interacts with and learns from the environment."""

	def learn(self, experiences):
		"""Update value parameters using given batch of experience tuples.

		Params
		======
			experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
			gamma (float): discount factor
		"""
		states, actions, rewards, next_states, dones = experiences

		best_actions = self.qnetwork_local(states).argmax(dim=1, keepdim=True)
		y_target = rewards + self.gamma * self.qnetwork_target(next_states).gather(1, best_actions) * (1 - dones)
		self.qnetwork_local.optimize(self.qnetwork_local(states).gather(1, actions), y_target.detach())

		self.soft_update(self.qnetwork_local, self.qnetwork_target)


class PrioExpReplayAgent(BaseAgent):

	def step(self, state, action, reward, next_state, done):
		# Save experience in replay memory
		probabilities = self.memory.compute_probs()
		self.memory.add(state, action, reward, next_state, done, 1.0 if len(probabilities) == 0 else np.max(probabilities))

		# Learn every UPDATE_EVERY time steps.
		self.t_step = (self.t_step + 1) % self.update_every
		if self.t_step == 0:
			# If enough samples are available in memory, get random subset and learn
			if len(self.memory) > self.batch_size:
				experiences = self.memory.sample()
				self.learn(experiences)

	def learn(self, experiences):
		"""Update value parameters using given batch of experience tuples.

		Params
		======
			experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done, index_of_sample) tuples
		"""
		(states, actions, rewards, next_states, dones, experiences_indices) = experiences

		y_target = rewards + self.gamma * self.qnetwork_target(next_states).max(dim=1, keepdim=True)[0] * (1 - dones)
		y_true = self.qnetwork_local(states).gather(1, actions)

		self.memory.update_probability((y_target-y_true).cpu().data.numpy().reshape(-1), experiences_indices)
		weights = self.memory.compute_weights(experiences_indices)

		self.qnetwork_local.optimize(y_true, y_target.detach(), torch.from_numpy(weights.reshape(-1, 1)).to(device))

		self.soft_update(self.qnetwork_local, self.qnetwork_target)
