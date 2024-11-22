import torch as th
import numpy as np
import dgl

class AccruedRewardReplayBuffer:
    """Replay buffer for multi-objective reinforcement learning with structured observation space using PyTorch tensors for observations and gate_features, NumPy for actions, rewards, done, and gate_sizes."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        rew_dim=2,
        max_size=10000,
        device='cpu',
    ):
        """Initialize the replay buffer with mixed data types (PyTorch for observations and gate_features, NumPy for actions, rewards, done, and gate_sizes)."""
        self.max_size = max_size
        self.ptr, self.size = 0, 0
        self.device = device
        # Initialize buffer for each part of the observation space, the original data are saved in CPU memory
        self.physical_image = th.zeros((max_size,) + obs_shape['physical_image'], dtype=th.float32, device='cpu')

        # Buffer for storing DGL graph objects (timing graph)
        self.timing_graph = [None] * max_size  # List for DGL graphs

        # # Initialize buffer for next observations
        # self.next_physical_image = th.zeros_like(self.physical_image)
        # self.next_timing_graph = [None] * max_size  # List for next DGL graphs

        # Initialize buffer for ht, ct, rewards, and done signals using NumPy arrays
        self.hts = th.zeros((max_size, 128), dtype=th.float32)
        self.cts = th.zeros((max_size, 128), dtype=th.float32)
        self.actions = np.zeros((max_size,) + action_shape, dtype=np.int32)
        self.rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.accrued_rewards = np.zeros((max_size, rew_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)  # Boolean array for done signals

    def add(self, obs, accrued_reward, prev_h_c, action, reward, done):
        """Add a new experience to the buffer, using mixed data types."""
        # Store current observation components
        self.physical_image[self.ptr] = obs['physical_image'].clone().to('cpu')
        self.timing_graph[self.ptr] = obs['timing_graph'].clone().to('cpu')

        # # Store next observation components
        # self.next_physical_image[self.ptr] = next_obs['physical_image'].clone().to('cpu')
        # self.next_timing_graph[self.ptr] = next_obs['timing_graph'].clone().to('cpu')

        # Store prev_h_c, reward, and done (NumPy arrays)
        self.hts[self.ptr] = prev_h_c[0].clone().to('cpu')
        self.cts[self.ptr] = prev_h_c[1].clone().to('cpu')
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.accrued_rewards[self.ptr] = np.array(accrued_reward).copy()
        self.dones[self.ptr] = np.array(done).copy()  # Store done as NumPy boolean array

        # Update buffer pointer and size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, replace=True, use_cer=False, device=None):
        """Sample a batch of experiences, batch DGL graphs, and return PyTorch tensors for observations and NumPy arrays for actions, rewards, and done."""
        inds = np.random.choice(self.size, batch_size, replace=replace)
        if use_cer:
            inds[0] = self.ptr - 1  # always use last experience for CER

        # Create a dictionary for observations (PyTorch tensors and NumPy arrays)
        observations = {
            'physical_image': self.physical_image[inds].clone().to(device),
        }

        # Create a dictionary for next observations
        # next_observations = {
        #     'physical_image': self.physical_image[inds].clone().to(device),
        # }

        # Batch the DGL graphs for both current and next observations
        batched_graph = dgl.batch([self.timing_graph[i].clone() for i in inds]).to(device)
        # next_batched_graph = dgl.batch([self.next_timing_graph[i].clone() for i in inds]).to(device)
        
        observations['timing_graph'] = batched_graph
        # next_observations['timing_graph'] = next_batched_graph

        # Convert actions, rewards, and done from NumPy to PyTorch tensors, if needed
        h_c_t = (self.hts[inds].clone().to(device), self.cts[inds].clone().to(device))
        actions = th.tensor(self.actions[inds], dtype=th.int32, device=device)
        rewards = th.tensor(self.rewards[inds], dtype=th.float32, device=device)
        accrued_rewards = th.tensor(self.accrued_rewards[inds], dtype=th.float32, device=device)
        dones = th.tensor(self.dones[inds], dtype=th.float32, device=device)

        experience_tuples = (
            observations,
            accrued_rewards,
            h_c_t,
            actions,
            rewards,
            dones,
        )

        return experience_tuples

    def cleanup(self):
        """Cleanup the buffer."""
        self.size, self.ptr = 0, 0
    
    def get_all_data(self, to_tensor=False, device=None):
        """Returns the whole buffer.

        Args:
            to_tensor: Whether to convert the data to tensors or not
            device: Device to use for the tensors

        Returns:
            Tuple of (obs, accrued_rewards, actions, rewards, next_obs, dones)
        """
        inds = np.arange(self.size)
        # Create a dictionary for observations (PyTorch tensors and NumPy arrays)
        observations = {
            'physical_image': self.physical_image[inds].clone().to(device),
        }

        # # Create a dictionary for next observations
        # next_observations = {
        #     'physical_image': self.physical_image[inds].clone().to(device),
        # }

        # Batch the DGL graphs for both current and next observations
        batched_graph = dgl.batch([self.timing_graph[i].clone() for i in inds]).to(device)
        # next_batched_graph = dgl.batch([self.next_timing_graph[i].clone() for i in inds]).to(device)
        
        observations['timing_graph'] = batched_graph
        # next_observations['timing_graph'] = next_batched_graph

        # Convert actions, rewards, and done from NumPy to PyTorch tensors, if needed
        # actions = th.tensor(self.actions[inds], dtype=th.float32, device=device)
        h_c_t = (self.hts[inds].clone().to(device), self.cts[inds].clone().to(device))
        actions = th.tensor(self.actions[inds], dtype=th.int32, device=device)
        rewards = th.tensor(self.rewards[inds], dtype=th.float32, device=device)
        accrued_rewards = th.tensor(self.accrued_rewards[inds], dtype=th.float32, device=device)
        dones = th.tensor(self.dones[inds], dtype=th.float32, device=device)

        experience_tuples = (
            observations,
            accrued_rewards,
            h_c_t,
            actions,
            rewards,
            dones,
        )
        
        return experience_tuples
    
    def __len__(self):
        """Get the size of the buffer."""
        return self.size

