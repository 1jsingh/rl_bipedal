import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
#%matplotlib inline

from ddpg_agent import Agent

# imports for rendering outputs in Jupyter.
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display

env = gym.make('BipedalWalker-v2')
env.seed(10)
agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=10)

agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))


# function to animate a list of frames
def animate_frames(frames):
    plt.figure(dpi = 72)
    plt.axis('off')

    # color option for plotting
    # use Greys for greyscale
    cmap = None if len(frames[0].shape)==3 else 'Greys'
    patch = plt.imshow(frames[0], cmap=cmap)  

    fanim = animation.FuncAnimation(plt.gcf(), \
        lambda x: patch.set_data(frames[x]), frames = len(frames), interval=30)
    
    display(display_animation(fanim, default_mode='once'))
    
frames = []
state = env.reset()
agent.reset()
for t in range(1000):
    action = agent.act(state)
    frames.append(env.render(mode='rgb_array')) 
    next_state, reward, done, _ = env.step(action)
    state=next_state
    if done:
        break

env.close()
animate_frames(frames)