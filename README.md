# About

Solving OpenAI's [Bipedal Walker](https://gym.openai.com/envs/BipedalWalker-v2/) environment using [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) algorithm. For comparison an implementation with [DDPG](https://arxiv.org/abs/1509.02971) algorithm is also provided.

### Results

![Trained Agent](images/trained_agent)


## Project Structure
2 different solutions have been implemented in this repo

* `ppo`: directory for ppo agent
  - `ppo_continuous.py`: code for running the ppo agent for the bipedal environment
  - `ppo_bipedal.ipynb`: jupyter notebook for agent training and visualisation
  - `utils.py`: utility functions
  - `deep_rl`: directory with modular functions for the PPO agent
* `ddpg`: directory for ddpg agent
  - `DDPG.ipynb`: jupyter notebook for training the agent
  - `ddpg_agent.py`: code for the agent model, experience replay and OU noise 
  - `model.py`: actor and critic networks
  
### Instructions

Open `ppo/ppo_bipedal.ipynb` to see an implementation of [PPO](https://arxiv.org/abs/1707.06347) with OpenAI Gym's BipedalWalker environment.
