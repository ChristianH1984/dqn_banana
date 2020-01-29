import numpy as np
import random
import torch
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseAgent():
	"""Interacts with and learns from the environment."""

	def __init__(self, qnetwork, memory):
		"""Initialize an Agent object.

		Params
		======
			state_size (int): dimension of each state
			action_size (int): dimension of each action
			seed (int): random seed
		"""
		# self.state_size = state_size
		# self.action_size = action_size
		# self.seed = random.seed(seed)

		# Q-Network
		self.qnetwork_local = qnetwork.to(device)
		self.qnetwork_target = qnetwork.clone().to(device)
		self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

		# Replay memory
		self.memory = memory

		# Initialize time step (for updating every UPDATE_EVERY steps)
		self.t_step = 0

	def step(self, state, action, reward, next_state, done):
		# Save experience in replay memory
		self.memory.add(state, action, reward, next_state, done)

		# Learn every UPDATE_EVERY time steps.
		self.t_step = (self.t_step + 1) % UPDATE_EVERY
		if self.t_step == 0:
			# If enough samples are available in memory, get random subset and learn
			if len(self.memory) > BATCH_SIZE:
				experiences = self.memory.sample()
				self.learn(experiences, GAMMA)

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

	def learn(self, experiences, gamma):
		"""Update value parameters using given batch of experience tuples.

		Params
		======
			experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
			gamma (float): discount factor
		"""
		pass

	def soft_update(self, local_model, target_model, tau):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target

		Params
		======
			local_model (PyTorch model): weights will be copied from
			target_model (PyTorch model): weights will be copied to
			tau (float): interpolation parameter
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class DQNAgent(BaseAgent):
	"""Interacts with and learns from the environment."""

	def learn(self, experiences, gamma):
		"""Update value parameters using given batch of experience tuples.

		Params
		======
			experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
			gamma (float): discount factor
		"""
		states, actions, rewards, next_states, dones = experiences
		y_target = rewards + gamma * self.qnetwork_target(next_states).max(dim=1, keepdim=True)[0] * (1 - dones)
		self.qnetwork_local.optimize(self.qnetwork_local(states).gather(1, actions), y_target.detach())

		self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


class DoubleDQNAgent(BaseAgent):
	"""Interacts with and learns from the environment."""

	def learn(self, experiences, gamma):
		"""Update value parameters using given batch of experience tuples.

		Params
		======
			experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
			gamma (float): discount factor
		"""
		states, actions, rewards, next_states, dones = experiences

		best_actions = self.qnetwork_local(states).argmax(dim=1, keepdim=True)
		y_target = rewards + gamma * self.qnetwork_target(next_states).gather(1, best_actions) * (1 - dones)
		self.qnetwork_local.optimize(self.qnetwork_local(states).gather(1, actions), y_target.detach())

		self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

class PrioExpReplayAgent(BaseAgent):

	def step(self, state, action, reward, next_state, done):
		# Save experience in replay memory
		state_torch = torch.from_numpy(state).float().unsqueeze(0).to(device)
		next_state_torch = torch.from_numpy(next_state).float().unsqueeze(0).to(device)

		#weight = reward + GAMMA * self.qnetwork_target(next_state_torch).max(dim=1, keepdim=True)[
		#	0] - self.qnetwork_target(state_torch)[:, action]
		#print("Agent weight", weight[0,0])
		probs = self.memory.compute_probs()
		if len(probs) == 0:
			weight = 1.0
		else:
			weight = np.max(probs)
		self.memory.add(state, action, reward, next_state, done, weight)

		# Learn every UPDATE_EVERY time steps.
		self.t_step = (self.t_step + 1) % UPDATE_EVERY
		if self.t_step == 0:
			# If enough samples are available in memory, get random subset and learn
			if len(self.memory) > BATCH_SIZE:
				experiences = self.memory.sample()
				self.learn(experiences, GAMMA)


	def learn(self, experiences, gamma):
		"""Update value parameters using given batch of experience tuples.

		Params
		======
			experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
			gamma (float): discount factor
		"""
		(states, actions, rewards, next_states, dones, weights, experiences_indices) = experiences

		#best_actions = self.qnetwork_local(states).argmax(dim=1, keepdim=True)
		#y_target = rewards + gamma * self.qnetwork_target(next_states).gather(1, best_actions) * (1 - dones)
		y_target = rewards + gamma * self.qnetwork_target(next_states).max(dim=1, keepdim=True)[0] * (1 - dones)
		y_true = self.qnetwork_local(states).gather(1, actions)

		self.memory.update_probability((y_target-y_true).cpu().data.numpy().reshape(-1), experiences_indices)
		weights = self.memory.compute_weights(experiences_indices)

		self.qnetwork_local.optimize(y_true, y_target.detach(), torch.from_numpy(weights.reshape(-1, 1)).to(device))

		#self.qnetwork_local.optimize(y_true, y_target.detach())

		self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
