from gymnasium.envs.registration import register
from mo_gymnasium.envs.rl_ctd import rl_ctd

register(
    id="rl-ctd-v0",
    entry_point="mo_gymnasium.envs.rl_ctd.rl_ctd:RL_CTD",
    kwargs={'current_design': 'mc_top'}
)