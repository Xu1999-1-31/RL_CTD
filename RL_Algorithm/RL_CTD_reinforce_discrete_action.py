import time
from copy import deepcopy
from typing import Callable, List, Optional, Union
from typing_extensions import override

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# from morl_baselines.common.accrued_reward_buffer import AccruedRewardReplayBuffer
from RL_CTD_Buffer import AccruedRewardReplayBuffer
import dgl
import wandb

from morl_baselines.common.evaluation import log_episode_info
from RL_Algorithm.RL_CTD_Morl_algorithm import MOPolicy
from morl_baselines.common.networks import mlp, layer_init
from morl_baselines.common.morl_algorithm import MOAgent

import os
import csv
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Global_var
import models
import scipy.optimize as opt

class PolicyNet(nn.Module):
    """Policy network."""

    def __init__(self, obs_shape, action_dim, rew_dim, net_arch):
        """Initialize the policy network.

        Args:
            obs_shape: Observation shape
            action_dim: Action dimension
            rew_dim: Reward dimension
            net_arch: Number of units per layer
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.rew_dim = rew_dim
        self.net_arch = net_arch

        # |S|+|R| -> ... -> |A|
        self.cnn = models.CNN() # output (batch_size, 64)
        self.gnn = models.MultiLayerTimingGNN(3, 64)
        self.lstm = nn.LSTMCell(128, 128)
        self.net = mlp(
            input_dim=128,
            output_dim=self.action_dim,
            net_arch=self.net_arch,
            activation_fn=nn.ReLU,
        )
        self.apply(layer_init)

    def forward(self, obs, prev_h_c):
        """Forward pass.

        Args:
            obs: Observation
            acc_reward: accrued reward

        Returns: Probability of each action

        """
        g = obs['timing_graph']
        I = obs['physical_image']
        
        with g.local_scope():
            g.ndata['nf'] = self.gnn(g)
            nf = dgl.mean_nodes(g, 'nf') # node feature
        lf = self.cnn(I) # layout feature
        input = th.cat([nf, lf], dim=1)
        h_t, c_t = self.lstm(input, prev_h_c)
        pi = self.net(h_t)
        # Normalized sigmoid
        # x_exp = th.sigmoid(pi)
        # probas = x_exp / th.sum(x_exp)
        probas = th.softmax(pi, dim=-1)
        return probas.view(-1, self.action_dim), (h_t, c_t)  # Batch Size x |Actions|

    def distribution(self, obs, prev_h_c):
        """Categorical distribution based on the action probabilities.

        Args:
            obs: observation
            acc_reward: accrued reward

        Returns: action distribution.

        """
        probas, (h_t, c_t) = self.forward(obs, prev_h_c)
        distribution = Categorical(probas)
        return distribution, (h_t, c_t)


class RL_CTD(MOPolicy, MOAgent):
    """Expected Utility Policy Gradient Algorithm.

    The idea is to condition the network on the accrued reward and to scalarize the rewards based on the episodic return (accrued + future rewards)
    Paper: D. Roijers, D. Steckelmacher, and A. Nowe, Multi-objective Reinforcement Learning for the Expected Utility of the Return. 2018.
    """

    def __init__(
        self,
        env: gym.Env,
        scalarization: Callable[[np.ndarray, np.ndarray], float],
        weights: np.ndarray = np.ones(2),
        id: Optional[int] = None,
        buffer_size: int = int(1e5),
        net_arch: List = [50],
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        project_name: str = "RL_CTD",
        experiment_name: str = "rl_ctd_v0",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        log_every: int = 1,
        device: Union[th.device, str] = "auto",
        seed: Optional[int] = None,
        parent_rng: Optional[np.random.Generator] = None,
        dynamic_weight_adjustment: bool = False,
    ):
        """Initialize the EUPG algorithm.

        Args:
            env: Environment
            scalarization: Scalarization function to use (can be non-linear)
            weights: Weights to use for the scalarization function
            id: Id of the agent (for logging)
            buffer_size: Size of the replay buffer
            net_arch: Number of units per layer
            gamma: Discount factor
            learning_rate: Learning rate (alpha)
            project_name: Name of the project (for logging)
            experiment_name: Name of the experiment (for logging)
            wandb_entity: Entity to use for wandb
            log: Whether to log or not
            log_every: Log every n episodes
            device: Device to use for NN. Can be "cpu", "cuda" or "auto".
            seed: Seed for the random number generator
            parent_rng: Parent random number generator (for reproducibility)
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

        self.env = env
        self.id = id
        # RL
        self.scalarization = scalarization
        self.weights = weights
        self.gamma = gamma
        
        # Learning
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.learning_rate = learning_rate
        self.buffer = AccruedRewardReplayBuffer(
            obs_shape=self.observation_shape,
            action_shape=self.action_shape,
            rew_dim=self.reward_dim,
            max_size=self.buffer_size,
        )
        self.net = PolicyNet(
            obs_shape=self.observation_shape,
            rew_dim=self.reward_dim,
            action_dim=self.action_dim,
            net_arch=self.net_arch,
        ).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log
        self.log_every = log_every
        if log and parent_rng is None:
            self.setup_wandb(self.project_name, self.experiment_name, wandb_entity)
            
        # dynamic weight adjustment
        self.dynamic_weight_adjustment = dynamic_weight_adjustment
        if self.dynamic_weight_adjustment:
            self.data_DRV = []
            self.data_TNS = []
            self.model_DRV = None
            self.model_TNS = None
            # Initialize other variables
            self.beta = 1.0  # Importance coefficient for TNS
            
    def report_time(self, elapsed_time, stage, file):
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        with open(file, 'a') as outfile:
            outfile.write(f'Runtime for {stage}: {hours} hours, {minutes} minutes, {seconds:.2f} seconds\n')

    def adjust_weights(self, f_DRV_current, omega_max=0.9, gamma=0.5):
        """Adjust weights dynamically based on the current DRV reward."""
        # Calculate new DRV weight
        omega_DRV_new = omega_max * (1 - f_DRV_current)
        # Smooth the weight adjustment
        omega_DRV = gamma * omega_DRV_new + (1 - gamma) * self.weights[0]
        # Ensure weights are within [0, omega_max]
        omega_DRV = np.clip(omega_DRV, 0, omega_max)
        omega_TNS = 1 - omega_DRV
        # Update weights
        self.weights = np.array([omega_DRV, omega_TNS])
    
    def hyperbolic_model(self, omega, A, a, b, c):
        """Hyperbolic model function."""
        exp_term = np.exp(a * (omega - b))
        return A * ((exp_term - 1) / (exp_term + 1)) + c
    
    def fit_improvement_model(self, data):
        """Fit the hyperbolic model to predict reward improvement.

        Args:
            data (list): List of (omega, delta_f) tuples.

        Returns:
            Callable: Function to predict delta_f given omega.
        """
        if len(data) < 4:
            # Not enough data to fit a nonlinear model
            return lambda omega: 0
        omegas, delta_fs = zip(*data)
        omegas = np.array(omegas)
        delta_fs = np.array(delta_fs)

        # Initial guess for parameters [A, a, b, c]
        p0 = [np.max(delta_fs), 1.0, np.mean(omegas), np.min(delta_fs)]
        try:
            params, _ = opt.curve_fit(hyperbolic_model, omegas, delta_fs, p0=p0, maxfev=10000)
            return lambda omega: hyperbolic_model(omega, *params)
        except RuntimeError:
            # Curve fitting failed; return zero improvement
            return lambda omega: 0

    def fit_improvement_model(self, data):
        """Fit the hyperbolic model to predict reward improvement.

        Args:
            data (list): List of (omega, delta_f) tuples.

        Returns:
            Callable: Function to predict delta_f given omega.
        """
        if len(data) < 4:
            # Not enough data to fit a nonlinear model
            return lambda omega: 0
        omegas, delta_fs = zip(*data)
        omegas = np.array(omegas)
        delta_fs = np.array(delta_fs)

        # Initial guess for parameters [A, a, b, c]
        p0 = [np.max(delta_fs), 1.0, np.mean(omegas), np.min(delta_fs)]
        try:
            params, _ = opt.curve_fit(self.hyperbolic_model, omegas, delta_fs, p0=p0, maxfev=10000)
            return lambda omega: self.hyperbolic_model(omega, *params)
        except RuntimeError:
            # Curve fitting failed; return zero improvement
            return lambda omega: 0

    def predict_reward_improvement(self, omega, model):
        """Predict reward improvement using the fitted hyperbolic model.

        Args:
            omega (float): Weight.
            model (Callable): Prediction model.

        Returns:
            float: Predicted reward improvement.
        """
        return model(omega)
    
    def __deepcopy__(self, memo):
        """Deep copy the policy."""
        copied_net = deepcopy(self.net)
        copied = type(self)(
            self.env,
            self.scalarization,
            self.weights,
            self.id,
            self.buffer_size,
            self.net_arch,
            self.gamma,
            self.learning_rate,
            self.project_name,
            self.experiment_name,
            log=self.log,
            device=self.device,
            parent_rng=self.parent_rng,
        )

        copied.global_step = self.global_step
        copied.optimizer = optim.Adam(copied_net.parameters(), lr=self.learning_rate)
        copied.buffer = deepcopy(self.buffer)
        return copied

    @override
    def get_policy_net(self) -> nn.Module:
        return self.net

    @override
    def get_buffer(self):
        return self.buffer

    @override
    def set_buffer(self, buffer):
        raise Exception("On-policy algorithms should not share buffer.")

    @override
    def set_weights(self, weights: np.ndarray):
        self.weights = weights

    @th.no_grad()
    @override
    def eval(self, obs: np.ndarray, accrued_reward: Optional[np.ndarray]) -> Union[int, np.ndarray]:
        if type(obs) is int:
            obs = th.as_tensor([obs]).to(self.device)
        else:
            obs = th.as_tensor(obs).to(self.device)
        accrued_reward = th.as_tensor(accrued_reward).float().to(self.device)
        return self.__choose_action(obs, accrued_reward)

    @th.no_grad()
    def __choose_action(self, obs, h_c_t) -> int:
        action, h_c_t = self.net.distribution(obs, h_c_t)
        action = action.sample().detach().item()
        return action, h_c_t

    @override
    def update(self):
        (
            obs,
            accrued_rewards,
            h_c_t,
            actions,
            rewards,
            terminateds,
        ) = self.buffer.get_all_data(to_tensor=True, device=self.device)

        episodic_return = th.sum(rewards, dim=0)
        scalarized_return = self.scalarization(episodic_return.cpu().numpy(), self.weights)
        scalarized_return = th.scalar_tensor(scalarized_return).to(self.device)

        discounted_forward_rewards = self._forward_cumulative_rewards(rewards)
        scalarized_values = self.scalarization(discounted_forward_rewards.cpu().numpy(), self.weights)
        scalarized_values = th.tensor(scalarized_values, device=self.device) 
        # For each sample in the batch, get the distribution over actions
        # final_reward = rewards[-1]  
        # scalarized_value = self.scalarization(final_reward.cpu().numpy(), self.weights)
        # scalarized_values = th.tensor(scalarized_value, device=self.device)

        current_distribution, _ = self.net.distribution(obs, h_c_t)
        # Policy gradient
        log_probs = current_distribution.log_prob(actions.squeeze())
        # scalarized_values = scalarized_values.expand_as(log_probs)
        # print(scalarized_values)
        loss = -th.mean(log_probs * scalarized_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.log:
            log_str = f"_{self.id}" if self.id is not None else ""
            wandb.log(
                {
                    f"losses{log_str}/loss": loss,
                    f"metrics{log_str}/scalarized_episodic_return": scalarized_return,
                    "global_step": self.global_step,
                },
            )

    def _forward_cumulative_rewards(self, rewards):
        flip_rewards = rewards.flip(dims=[0])
        cumulative_rewards = th.zeros(self.reward_dim).to(self.device)
        for i in range(len(rewards)):
            cumulative_rewards = self.gamma * cumulative_rewards + flip_rewards[i]
            flip_rewards[i] = cumulative_rewards
        forward_rewards = flip_rewards.flip(dims=[0])
        return forward_rewards

    
    def train(self, total_timesteps: int, eval_env: Optional[gym.Env] = None, eval_freq: int = 1000, start_time=None):
        """Train the agent.

        Args:
            total_timesteps: Number of timesteps to train for
            eval_env: Environment to run policy evaluation on
            eval_freq: Frequency of policy evaluation
            start_time: Start time of the training (for SPS)
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
        obs_list = self.env.reset()
        accrued_reward_tensor = th.zeros(self.reward_dim, dtype=th.float32).float().to(self.device)
        prev_h_c = (th.zeros((1, 128), dtype=th.float32).to(self.device), th.zeros((1, 128), dtype=th.float32).to(self.device))

        if self.dynamic_weight_adjustment:
            # Initialize previous rewards
            prev_f_DRV = 0.0
            prev_f_TNS = 0.0
            # Initialize parameters for weight adjustment
            omega_max = 0.7
            gamma = 0.5
            delta_range = 0.2
            num_candidates = 10  
        
        
        # Training loop
        for _ in range(1, total_timesteps + 1):
            self.global_step += 1
            action_list = []
            for i in range(len(obs_list)):
                obs = obs_list[i]
                th_obs = {}
                for key, value in obs.items():
                    if isinstance(value, th.Tensor):
                        th_obs[key] = value.to(self.device).unsqueeze(0)
                    else:
                        th_obs[key] = value.to(self.device)
                with th.no_grad():
                    # For training, takes action according to the policy
                    action, h_c_t = self.__choose_action(th_obs, prev_h_c)
                    action_list.append(action)
                if i == len(obs_list) - 1:
                    terminated = True
                    vec_reward, (wns, tns, drc) = self.env.step(action_list)
                    file_exists = os.path.isfile(csv_filename) 
                    with open(csv_filename, mode='a', newline='') as file: 
                            writer = csv.writer(file) 
                            if not file_exists: 
                                writer.writerow(["action", "WNS", "TNS", "DRC", "Reward"]) # Write header if file doesn't exist 
                            writer.writerow([action_list, wns, tns, drc, vec_reward[0] + vec_reward[1]])
                else:
                    terminated = False
                    vec_reward= np.array([0, 0])
            
                # Memory update
                self.buffer.add(obs, accrued_reward_tensor.cpu().numpy(), prev_h_c, action, vec_reward, terminated)
                prev_h_c = h_c_t
                accrued_reward_tensor += th.from_numpy(vec_reward).to(self.device)
                
                        
                if eval_env is not None and self.log and self.global_step % eval_freq == 0:
                    self.policy_eval_esr(eval_env, scalarization=self.scalarization, weights=self.weights, log=self.log)
                
                if terminated:
                    if self.dynamic_weight_adjustment:
                        # Update data for models
                        f_DRV_current = accrued_reward_tensor.cpu().numpy()[0]  # Assuming DRV is the first reward
                        f_TNS_current = accrued_reward_tensor.cpu().numpy()[1]  # Assuming TNS is the second reward
                        delta_f_DRV = f_DRV_current - prev_f_DRV
                        delta_f_TNS = f_TNS_current - prev_f_TNS
                        self.data_DRV.append((self.weights[0], delta_f_DRV))
                        self.data_TNS.append((self.weights[1], delta_f_TNS))
                        prev_f_DRV = f_DRV_current
                        prev_f_TNS = f_TNS_current
                        
                        # Fit improvement models
                        self.model_DRV = self.fit_improvement_model(self.data_DRV)
                        self.model_TNS = self.fit_improvement_model(self.data_TNS)
                        
                        # Adjust weights based on current DRV reward
                        self.adjust_weights(f_DRV_current, omega_max, gamma)
                        
                        # Optimize weights using prediction models
                        best_Q = -np.inf
                        for delta in np.linspace(-delta_range, delta_range, num_candidates):
                            omega_DRV_candidate = self.weights[0] + delta
                            omega_TNS_candidate = 1 - omega_DRV_candidate
                            # Ensure weights are within valid range
                            if 0 <= omega_DRV_candidate <= omega_max:
                                delta_f_DRV = self.predict_reward_improvement(omega_DRV_candidate, self.model_DRV)
                                delta_f_TNS = self.predict_reward_improvement(omega_TNS_candidate, self.model_TNS)
                                Q = delta_f_DRV + self.beta * delta_f_TNS
                                if Q > best_Q:
                                    best_Q = Q
                                    omega_DRV_opt = omega_DRV_candidate
                                    omega_TNS_opt = omega_TNS_candidate
                        
                        # Update weights with optimized values
                        self.weights = np.array([omega_DRV_opt, omega_TNS_opt])
                    
                    # NN is updated at the end of each episode
                    self.update()
                    self.buffer.cleanup()
                    self.num_episodes += 1
                    reward = th.sum(accrued_reward_tensor).item()
                    accrued_reward_tensor = th.zeros(self.reward_dim).float().to(self.device)
                    prev_h_c = (th.zeros((1, 128), dtype=th.float32).to(self.device), th.zeros((1, 128), dtype=th.float32).to(self.device))

                    
                    if self.log and self.num_episodes % self.log_every == 0:
                        log_str = f"_{self.id}" if self.id is not None else ""
                        to_log = {f"TNS{log_str}/tns": tns, f"DRC{log_str}/drc": drc, f"Reward{log_str}/Reward": reward}
                        wandb.log(to_log)

            if self.log and self.global_step % 5 == 0:
                print("SPS:", int(self.global_step / (time.time() - start_time)))
                wandb.log({"charts/SPS": int(self.global_step / (time.time() - start_time)), "global_step": self.global_step})
        
            time_for_step = time.time() - end_time
            end_time = time.time()
            runtime = time.time() - start_time
            self.report_time(time_for_step, 'current step', runtime_filename)
            self.report_time(runtime, 'total step', runtime_filename)
        

    @override
    def get_config(self) -> dict:
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
            "seed": self.seed,
        }
