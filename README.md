[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135608-be87357e-7d12-11e8-8eca-e6d5fabdba6b.gif "Trained Agent"


# Actor-Critic Methods

### Instructions

Open `DDPG.ipynb` to see an implementation of DDPG with OpenAI Gym's BipedalWalker environment.

### Results

![Trained Agent][image1]


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
