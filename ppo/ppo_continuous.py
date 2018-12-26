#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *

def ppo_continuous(name,num_workers=1):

    single_process = (num_workers==1)
    config = Config()
    log_dir = get_default_log_dir(ppo_continuous.__name__)
    config.task_fn = lambda: Task(name,num_envs=num_workers,single_process=single_process)
    config.eval_env = Task(name, log_dir=log_dir)

    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, gate=F.tanh),
        critic_body=FCBody(config.state_dim, gate=F.tanh))
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 143360
    config.max_steps = 1e6
    config.num_workers = num_workers
    config.state_normalizer = MeanStdNormalizer()
    config.logger = get_logger()
    agent = run_steps(PPOAgent(config))
    return agent