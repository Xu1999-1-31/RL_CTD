import time
from copy import deepcopy
from typing import Callable, List, Optional, Union
from typing_extensions import override

import gymnasium as gym
import numpy as np
import torch as th

from RL_CTD_Morl_algorithm import MOPolicy
from morl_baselines.common.morl_algorithm import MOAgent

import os
import csv
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import Global_var
# import models


class RL_CTD_SA(MOPolicy, MOAgent):
    def __init__(
        self,
        env: gym.Env,
        id: Optional[int] = None,
        project_name: str = "RL_CTD_SA",
        experiment_name: str = "rl_ctd_sa_v0",
        log: bool = True,
        log_every: int = 1,
        device: Union[th.device, str] = "auto",
        seed: Optional[int] = None,
        parent_rng: Optional[np.random.Generator] = None,
    ):
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

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log
        self.log_every = log_every
            
    def report_time(self, elapsed_time, stage, file):
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60
        with open(file, 'a') as outfile:
            outfile.write(f'Runtime for {stage}: {hours} hours, {minutes} minutes, {seconds:.2f} seconds\n')

    # @th.no_grad()
    @override
    def eval(self, obs: np.ndarray, accrued_reward: Optional[np.ndarray]) -> Union[int, np.ndarray]:
        pass

    # @override
    def update(self):
        pass
    
    def train(self, total_timesteps: int, initial_temp=100, cooling_rate=0.95, eval_env: Optional[gym.Env] = None, start_time=None):
        """Train the agent using Simulated Annealing.

        Args:
            total_timesteps: Number of timesteps to run for
            initial_temp: Starting temperature for simulated annealing
            cooling_rate: Rate at which the temperature cools down
            eval_env: Environment to run policy evaluation on
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
            
        temp = initial_temp

        action_list = [np.random.choice([0, 1, 2]) for _ in range(len(self.env.reset()))]
        best_action_list = action_list.copy()
        
        # Evaluate the initial action list
        vec_reward, (wns, tns, drc) = self.env.step(action_list)
        best_reward = np.sum(vec_reward) 

        
        for step in range(1, total_timesteps + 1):
            # Determine the number of elements to modify based on the current temperature
            num_updates = max(1, int(len(action_list) * (temp / initial_temp)))  # Fewer updates as temperature decreases

            # Generate a new candidate action list
            new_action_list = action_list.copy()
            indices_to_modify = np.random.choice(len(action_list), num_updates, replace=False)
            for idx in indices_to_modify:
                new_action_list[idx] = np.random.choice([0, 1, 2])

            # Evaluate the new candidate solution
            new_vec_reward, (new_wns, new_tns, new_drc) = self.env.step(new_action_list)
            new_reward = np.sum(new_vec_reward)  # Scaling each reward component by 0.5 and summing

            # Calculate acceptance probability
            delta_reward = new_reward - best_reward
            acceptance_probability = np.exp(delta_reward / temp) if delta_reward < 0 else 1


            # Update the best solution if the new reward is higher
            if new_reward > best_reward:
                action_list = new_action_list
                best_reward = new_reward
                wns, tns, drc = new_wns, new_tns, new_drc
            # Accept the new solution based on acceptance probability
            elif np.random.rand() < acceptance_probability:
                action_list = new_action_list

            # Logging rewards and actions
            file_exists = os.path.isfile(csv_filename)
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow([
                        "action_list", 
                        "new_WNS", "new_TNS", "new_DRC", "new_Reward", 
                        "best_WNS", "best_TNS", "best_DRC", "best_Reward"
                    ])
                writer.writerow([
                    new_action_list, 
                    new_wns, new_tns, new_drc, new_reward,  # Log new values
                    wns, tns, drc, best_reward           # Log best values
                ])

            # Reduce the temperature
            temp *= cooling_rate

            # Stop if temperature is too low
            if temp < 1e-5:
                break

            # Timing and logging
            time_for_step = time.time() - end_time
            end_time = time.time()
            runtime = time.time() - start_time
            self.report_time(time_for_step, 'current step', "runtime")
            self.report_time(runtime, 'total step', "runtime")

        # Return the best action list and its reward
        return best_action_list, best_reward
        

    @override
    def get_config(self) -> dict:
        return {
            "env_id": self.env.unwrapped.spec.id,
            "seed": self.seed,
        }
