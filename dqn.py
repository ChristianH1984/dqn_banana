import numpy as np


class DQN():
	def __init__(self, env, agent):
		self.env = env
		self.agent = agent

	def train(self, eps_iterator, max_iter):
		brain_name = self.env.brain_names[0]
		brain = self.env.brains[brain_name]
		scores = []

		for i in range(max_iter):
			score = 0.0
			env_info = self.env.reset(train_mode=True)[brain_name]
			state = env_info.vector_observations[0]
			done = False
			eps = next(eps_iterator)
			while not done:
				action = self.agent.act(state, brain.vector_action_space_size, eps)
				env_info = self.env.step(action)[brain_name]
				next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
				self.agent.step(state, action, reward, next_state, done)
				state = next_state
				score += reward

			scores.append(score)

			print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores[-20:])), end="")
			if i % 100 == 0:
				print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores[-20:])))

		return scores
