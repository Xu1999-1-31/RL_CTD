from gymnasium.envs.registration import register
from mo_gymnasium.envs.rl_ptd2 import rl_ptd

register(
    id="rl-ptd-v2",
    entry_point="mo_gymnasium.envs.rl_ptd2.rl_ptd:RL_PTD",
    kwargs={'current_design': 'aes_cipher_top'}
)