import numpy as np
from RL_CTD_reinforce_discrete_action import RL_CTD

#'''

def scalarization(reward, weights):
    return np.dot(reward, weights)

def create_agent(env):
    agent = RL_CTD(
        env=env,
        scalarization=scalarization,
        weights = np.array([0.5, 0.5]), # weights of reward values
        buffer_size = int(1000), # size of resampling buffer 
        gamma = 0.95, # discount factor
        learning_rate=1e-3,  # 学习率
        net_arch = [256, 256],
        project_name="RL_CTD",
        experiment_name="rl_ctd_v0",
        log=False,  # use wandb to log
        log_every=1,
        device="auto",
        seed=42,  # random seed
    )
    return agent

# '''
