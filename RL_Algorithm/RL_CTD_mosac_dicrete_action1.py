"""Multi-objective Soft Actor-Critic (SAC) algorithm for discrete action spaces.

It implements a multi-objective critic with weighted sum scalarization.
The implementation of this file is largely based on CleanRL's SAC implementation
https://github.com/vwxyzjn/cleanrl/blob/28fd178ca182bd83c75ed0d49d52e235ca6cdc88/cleanrl/sac_continuous_action.py
"""

import time
from copy import deepcopy
from typing import Optional, Tuple, Union
from typing_extensions import override

import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.evaluation import log_episode_info
from morl_baselines.common.morl_algorithm import MOPolicy
from morl_baselines.common.networks import mlp, polyak_update
from morl_baselines.common.morl_algorithm import MOAgent

import os
import csv
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ALGO LOGIC: initialize agent here:
class MOSoftQNetwork(nn.Module):
    """Soft Q-network: S, A -> ... -> |R| (multi-objective)."""

    def __init__(
        self,
        obs_shape,
        action_dim,
        reward_dim,
        net_arch=[256, 256],
    ):
        """Initialize the soft Q-network."""
        super().__init__()
        self.obs_shape = obs_shape # row * column
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.net_arch = net_arch

        # S, A -> ... -> |R| (multi-objective)
        self.critic = mlp(
            input_dim=self.obs_shape,
            output_dim=self.action_dim * self.reward_dim,
            net_arch=self.net_arch,
            activation_fn=nn.ReLU,
        )


    def forward(self, x):
        """Forward pass of the soft Q-network."""
        # q_values = self.critic(g, img, padding_mask, node_num, Gate_num)
        q_values = self.critic(x)
        return q_values.view(-1, self.action_dim, self.reward_dim)



class MOSACActor(nn.Module):
    """Actor network: S -> A. Does not need any multi-objective concept."""

    def __init__(
        self,
        obs_shape: int,
        action_dim: int,
        reward_dim: int,
        net_arch=[256, 256],
    ):
        """Initialize SAC actor."""
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.net_arch = net_arch

        # S -> ... -> |A| (mean)
        #          -> |A| (std)
        self.fc1 = nn.Linear(self.obs_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_logits = nn.Linear(256, self.action_dim)

    def forward(self, x):
        """Forward pass of the actor network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        """Get action from the actor network."""
        logits = self(x)
        action_probs = F.softmax(logits, dim = -1) # action distribution
        action_dist = th.distributions.Categorical(action_probs)
        action = action_dist.sample().view(-1, 1)
        z = (action_probs == 0.0).float() * 1e-8
        log_probs = th.log(action_probs + z)
        return action, log_probs, action_probs


class RL_PTD(MOPolicy, MOAgent):
    """Multi-objective Soft Actor-Critic (SAC) algorithm.

    It is a multi-objective version of the SAC algorithm, with multi-objective critic and weighted sum scalarization.
    """

    def __init__(
        self,
        env: gym.Env,
        weights: np.ndarray,
        scalarization=th.matmul,
        buffer_size: int = int(1e6),
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 128,
        learning_starts: int = int(1e3),
        net_arch=[256, 256],
        policy_lr: float = 3e-4,
        q_lr: float = 1e-3,
        a_lr: float = 1e-3,
        project_name: str = "RL_PTD",
        experiment_name: str = "rl_ptd_v1",
        policy_freq: int = 2,
        target_net_freq: int = 1,
        alpha: float = 0.2,
        autotune: bool = True,
        id: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        log_every: int = 1000,
        seed: int = 42,
        parent_rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the MOSAC algorithm.

        Args:
            env: Env
            weights: weights for the scalarization
            scalarization: scalarization function
            buffer_size: buffer size
            gamma: discount factor
            tau: target smoothing coefficient (polyak update)
            batch_size: batch size
            learning_starts: how many steps to collect before triggering the learning
            net_arch: number of nodes in the hidden layers
            policy_lr: learning rate of the policy
            q_lr: learning rate of the q networks
            policy_freq: the frequency of training policy (delayed)
            target_net_freq: the frequency of updates for the target networks
            alpha: Entropy regularization coefficient
            autotune: automatic tuning of alpha
            id: id of the SAC policy, for multi-policy algos
            device: torch device
            torch_deterministic: whether to use deterministic version of pytorch
            log: logging activated or not
            seed: seed for the random generators
            parent_rng: parent random generator, for multi-policy algos
        """
        MOAgent.__init__(self, env, device, seed=seed)
        MOPolicy.__init__(self, None, device)
        # Seeding
        self.seed = seed
        self.parent_rng = parent_rng
        if parent_rng is not None:
            self.np_random = parent_rng
        else:
            self.np_random = np.random.default_rng(self.seed)

        # env setup
        self.env = env
        self.obs_shape = self.observation_shape

        # Scalarization
        self.weights = weights
        self.weights_tensor = th.from_numpy(self.weights).float().to(self.device)
        self.batch_size = batch_size
        self.scalarization = scalarization

        # SAC Parameters
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.learning_starts = learning_starts
        self.net_arch = net_arch
        self.policy_lr = policy_lr
        self.learning_rate = policy_lr
        self.q_lr = q_lr
        self.a_lr = a_lr
        self.policy_freq = policy_freq
        self.target_net_freq = target_net_freq

        # Networks
        self.actor = MOSACActor(
            obs_shape=self.obs_shape[0],
            action_dim=self.action_dim,
            reward_dim=self.reward_dim,
            net_arch=self.net_arch,
        ).to(self.device)

        self.qf1 = MOSoftQNetwork(
            obs_shape=self.obs_shape[0], action_dim=self.action_dim, reward_dim=self.reward_dim, net_arch=self.net_arch
        ).to(self.device)
        self.qf2 = MOSoftQNetwork(
            obs_shape=self.obs_shape[0], action_dim=self.action_dim, reward_dim=self.reward_dim, net_arch=self.net_arch
        ).to(self.device)
        self.qf1_target = MOSoftQNetwork(
            obs_shape=self.obs_shape[0], action_dim=self.action_dim, reward_dim=self.reward_dim, net_arch=self.net_arch
        ).to(self.device)
        self.qf2_target = MOSoftQNetwork(
            obs_shape=self.obs_shape[0], action_dim=self.action_dim, reward_dim=self.reward_dim, net_arch=self.net_arch
        ).to(self.device)
        self.qf1_target.requires_grad_(False)
        self.qf2_target.requires_grad_(False)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.policy_lr)

        # Automatic entropy tuning
        self.autotune = autotune
        if self.autotune:
            #self.target_entropy =  -np.log(1/self.action_dim)
            self.target_entropy =  0.98*-np.log(1.0/self.action_dim)
            self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
            #self.log_alpha = th.tensor([-1.0], requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.a_lr)
        else:
            self.alpha = alpha
        self.alpha_tensor = th.tensor([self.alpha]).to(self.device)
        
        # Buffer
        self.env.observation_space.dtype = np.float32
        self.buffer = ReplayBuffer(
            obs_shape=self.obs_shape,
            action_dim=self.action_shape[0],
            rew_dim=self.reward_dim,
            max_size=self.buffer_size,
        )

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log
        self.log_every = log_every
        if log and parent_rng is None:
            self.setup_wandb(self.project_name, self.experiment_name, wandb_entity)
            
    def report_time(self, elapsed_time, stage, file):
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        with open(file, 'a') as outfile:
            outfile.write(f'Runtime for {stage}: {hours} hours, {minutes} minutes, {seconds:.2f} seconds\n')

    def get_config(self) -> dict:
        """Returns the configuration of the policy."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "buffer_size": self.buffer_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "net_arch": self.net_arch,
            "policy_lr": self.policy_lr,
            "q_lr": self.q_lr,
            "policy_freq": self.policy_freq,
            "target_net_freq": self.target_net_freq,
            "alpha": self.alpha,
            "autotune": self.autotune,
            "seed": self.seed,
        }

    def __deepcopy__(self, memo):
        """Deep copy of the policy.

        Args:
            memo (dict): memoization dict
        """
        copied = type(self)(
            env=self.env,
            weights=self.weights,
            scalarization=self.scalarization,
            buffer_size=self.buffer_size,
            gamma=self.gamma,
            tau=self.tau,
            batch_size=self.batch_size,
            learning_starts=self.learning_starts,
            net_arch=self.net_arch,
            policy_lr=self.policy_lr,
            q_lr=self.q_lr,
            a_lr=self.a_lr,
            policy_freq=self.policy_freq,
            target_net_freq=self.target_net_freq,
            alpha=self.alpha,
            autotune=self.autotune,
            id=self.id,
            device=self.device,
            log=self.log,
            seed=self.seed,
            parent_rng=self.parent_rng,
            project_name = self.project_name,
            experiment_name = self.experiment_name
        )

        # Copying networks
        copied.actor = deepcopy(self.actor)
        copied.qf1 = deepcopy(self.qf1)
        copied.qf2 = deepcopy(self.qf2)
        copied.qf1_target = deepcopy(self.qf1_target)
        copied.qf2_target = deepcopy(self.qf2_target)

        copied.global_step = self.global_step
        copied.actor_optimizer = optim.Adam(copied.actor.parameters(), lr=self.policy_lr, eps=1e-5)
        copied.q_optimizer = optim.Adam(list(copied.qf1.parameters()) + list(copied.qf2.parameters()), lr=self.q_lr)
        if self.autotune:
            copied.a_optimizer = optim.Adam([copied.log_alpha], lr=self.a_lr)
        copied.alpha_tensor = th.scalar_tensor(copied.alpha).to(self.device)
        copied.buffer = deepcopy(self.buffer)
        return copied

    @override
    def get_buffer(self):
        return self.buffer

    @override
    def set_buffer(self, buffer):
        self.buffer = buffer

    @override
    def get_policy_net(self) -> th.nn.Module:
        return self.actor

    @override
    def set_weights(self, weights: np.ndarray):
        self.weights = weights
        self.weights_tensor = th.from_numpy(self.weights).float().to(self.device)

    @override
    def eval(self, obs: np.ndarray, w: Optional[np.ndarray] = None) -> Union[int, np.ndarray]:
        """Returns the best action to perform for the given obs.

        Args:
            obs: observation as a numpy array
            w: None
        Return:
            action as a numpy array (continuous actions)
        """
        obs = th.as_tensor(obs).float().to(self.device)
        obs = obs.unsqueeze(0)
        with th.no_grad():
            action, _, _ = self.actor.get_action(obs)

        return action[0].detach().cpu().numpy()

    @override
    def update(self):
        (mb_obs, mb_act, mb_rewards, mb_next_obs, mb_dones) = self.buffer.sample(
            self.batch_size, to_tensor=True, device=self.device
        )#(128,20)
        with th.no_grad():
            _, log_probs, action_probs = self.actor.get_action(mb_next_obs)
            # (!) Q values are scalarized before being compared (min of ensemble networks)

            qf1_next_target = (self.qf1_target(mb_next_obs) * self.weights_tensor).sum(dim = -1)#(128,9,2) (1,1,2)这里要沿着最后一个维度相乘
            qf2_next_target = (self.qf2_target(mb_next_obs) * self.weights_tensor).sum(dim = -1)#(128,9)

            
            soft_state_values = (action_probs * (th.min(qf1_next_target, qf2_next_target) - self.alpha_tensor * log_probs)).sum(dim = 1)#(128,9)
            scalarized_rewards = (mb_rewards * self.weights_tensor).sum(dim = -1)
            next_q_value = scalarized_rewards.flatten() + (1 - mb_dones.flatten()) * self.gamma * soft_state_values

        qf1_a_values = (self.qf1(mb_obs) * self.weights_tensor).sum(dim = -1).gather(1, mb_act.long()).squeeze(-1)#(128,9)用(128,9)gather
        qf2_a_values = (self.qf2(mb_obs) * self.weights_tensor).sum(dim = -1).gather(1, mb_act.long()).squeeze(-1)

        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)#两个都是(128,)的标量
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()
        self.q_optimizer.step()

        if self.global_step % self.policy_freq == 0:  # TD 3 Delayed update support
            for _ in range(self.policy_freq):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                _, log_probs, action_probs = self.actor.get_action(mb_obs)
                entropies = th.sum(action_probs * log_probs, dim=1)
                # (!) Q values are scalarized before being compared (min of ensemble networks)
                qf1_values = (self.qf1(mb_obs) * self.weights_tensor).sum(dim = -1)
                qf2_values = (self.qf2(mb_obs) * self.weights_tensor).sum(dim = -1)
                min_qf_values = th.min(qf1_values, qf2_values)
                actor_loss = (action_probs * (self.alpha_tensor * log_probs - min_qf_values)).sum(dim = 1).mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune:
                    #print(f'action_probs: {action_probs}##{action_probs.shape}')
                    #print(f'log_probs: {log_probs}##{log_probs.shape}')
                    #print(f'entopies: {entropies}##{entropies.shape}')
                    #print(f'target_entropy: {self.target_entropy}')
                    alpha_loss = (self.log_alpha * (self.target_entropy + entropies).detach()).mean()
                    #print(f'alpha_loss :{alpha_loss}')
                    #alpha_loss = (-self.log_alpha * (log_probs + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha_tensor = self.log_alpha.exp()
                    self.alpha = max(self.log_alpha.exp().item(), 0.01)

        # update the target networks
        if self.global_step % self.target_net_freq == 0:
            polyak_update(params=self.qf1.parameters(), target_params=self.qf1_target.parameters(), tau=self.tau)
            polyak_update(params=self.qf2.parameters(), target_params=self.qf2_target.parameters(), tau=self.tau)
            self.qf1_target.requires_grad_(False)
            self.qf2_target.requires_grad_(False)

        if self.global_step % 100 == 0 and self.log:
            log_str = f"_{self.id}" if self.id is not None else ""
            to_log = {
                f"losses{log_str}/alpha": self.alpha,
                f"losses{log_str}/qf1_values": qf1_a_values.mean().item(),
                f"losses{log_str}/qf2_values": qf2_a_values.mean().item(),
                f"losses{log_str}/qf1_loss": qf1_loss.item(),
                f"losses{log_str}/qf2_loss": qf2_loss.item(),
                #f"losses{log_str}/qf_loss": qf_loss.item() / 2.0,
                f"losses{log_str}/actor_loss": actor_loss.item(),
                "global_step": self.global_step,
            }
            if self.autotune:
                to_log[f"losses{log_str}/alpha_loss"] = alpha_loss.item()
            wandb.log(to_log)


    def train(self, total_timesteps: int, eval_env: Optional[gym.Env] = None, start_time=None, reward_log=True):
        """Train the agent.

        Args:
            total_timesteps (int): Total number of timesteps (env steps) to train for
            eval_env (Optional[gym.Env]): Gym environment used for evaluation.
            start_time (Optional[float]): Starting time for the training procedure. If None, it will be set to the current time.
        """
        if start_time is None:
            start_time = time.time()
        end_time = time.time()
        
        csv_filename = f"rewards.csv" 
        if os.path.isfile(csv_filename):
            os.remove(csv_filename) 
        
        runtime_filename = f"runtime" 
        if os.path.isfile(runtime_filename):
            os.remove(runtime_filename) 

        # TRY NOT TO MODIFY: start the game
        obs = self.env.reset()
        for step in range(total_timesteps):
            # ALGO LOGIC: put action logic here
            if self.global_step < self.learning_starts:
                actions = self.env.action_space.sample()
            else:
                th_obs = th.as_tensor(obs).float().to(self.device)
                th_obs = th_obs.unsqueeze(0)
                actions, _, _ = self.actor.get_action(th_obs)
                actions = actions.detach().cpu().numpy()

            
            next_obs, rewards, terminated, state_vector = self.env.step(actions)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs
            self.buffer.add(obs=obs, next_obs=real_next_obs, action=actions, reward=rewards, done=terminated)
            
            #print(rewards)
            if reward_log == True and state_vector[2] != None and self.log:
                log_str = f"_{self.id}" if self.id is not None else ""
                to_log = {f"TNS{log_str}/TNS": state_vector[1], f"DRC{log_str}/DRC": state_vector[2]}
                wandb.log(to_log)
           
            if terminated == True:
                rewards_to_log = rewards[0] + rewards[1] # Convert tensor to numpy array for easier handling 
                file_exists = os.path.isfile(csv_filename)
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow(["Reward", "Total Reward", "(WNS, TNS, DRC)"]) # Write header if file doesn't exist 
                    writer.writerow([rewards, rewards[0] + rewards[1], state_vector])


            if terminated == True and reward_log == True and self.log:
                # Log rewards to a CSV file 
                rewards_to_log = 1/2*rewards[0] + 1/2*rewards[1] # Convert tensor to numpy array for easier handling 
                file_exists = os.path.isfile(csv_filename)
                with open(csv_filename, mode='a', newline='') as file: 
                    writer = csv.writer(file)  
                    if not file_exists: 
                        writer.writerow(["Reward", "Total Reward", "(WNS, TNS, DRC)"]) # Write header if file doesn't exist 
                    writer.writerow([rewards, rewards[0] + rewards[1], state_vector])
                log_str = f"_{self.id}" if self.id is not None else ""
                print(rewards_to_log)
                to_log = {f"Reward{log_str}/Reward": rewards_to_log}
                wandb.log(to_log)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            if terminated:
                obs = self.env.reset()
                    
            # ALGO LOGIC: training.
            if self.global_step > self.learning_starts:
                self.update()
                if self.log and self.global_step % 100 == 0:
                    print("SPS:", int(self.global_step / (time.time() - start_time)))
                    wandb.log(
                        {"charts/SPS": int(self.global_step / (time.time() - start_time)), "global_step": self.global_step}
                    )

            time_for_step = time.time() - end_time
            end_time = time.time()
            runtime = time.time() - start_time
            self.report_time(time_for_step, 'current step', runtime_filename)
            self.report_time(runtime, 'total step', runtime_filename)
            self.global_step += 1
