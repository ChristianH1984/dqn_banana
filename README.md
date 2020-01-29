# Deep Q-Learning to solve Unity-Banana problem

The repository provides three implementations of deep Q-learning Networks (DQN): plain vanilla DQN, double DQN and 
Prioritized Experience Replay DQN. Compared to the vanilla implementation the double DQN algorithm uses different networks
for selection of the best action and the estimation of the expected Q-value. This helps to reduce the effects of 
overestimation during early training stages. With the help of prioritized experience replays samples which lead to a 
larger change in the Q-table and are therefore more significant are used more frequently and thus leads to a more
 efficient training.
  
 ![Alt text](unity-banana.png?raw=true "Title")
 
The agent is supposed the solve the Unity-Banana problem: collecting as many yellow bananas as possible while avoiding any
collision with blue bananas. The agent receives a reward of +1 for each yellow and a penalty of -1 for each blue banana.
During each step he receives a vector consisting of the agent's velocity and ray-based perception of objects. 
The following actions are possible: forward, backward, turn left, turn right.
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 
consecutive episodes.

## Installation
Clone the github repo
- git clone https://github.com/ChristianH1984/dqn_banana.git
- cd dqn_banana
- conda env create --name dqn_banana --file=environment.yml
- activate dqn_banana
- open Navigation.ipynb and have fun :-)