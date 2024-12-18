import argparse
import mo_gymnasium as mo_gym
import RL_CTD
import tee

parser = argparse.ArgumentParser(description='This is the main python program for RL_CTD project.')

parser.add_argument('--train', action='store_false', help='Train the model')
parser.add_argument('--stdout', type=str, help='File path to redirect stdout')
parser.add_argument('--stderr', type=str, help='File path to redirect stderr')
parser.add_argument('--disable-log', action='store_true', help='Disable logging to file')


args = parser.parse_args()

loggers = [] # logger to store iostream

if not args.disable_log:
    # set stdout
    if args.stdout:
        stdout_file = args.stdout
    else:
        stdout_file = 'stdout.log'  # 默认文件名
    
    # redirect stdout
    stdout_logger = tee.StdoutTee(stdout_file, mode="w")
    loggers.append(stdout_logger)

    # set stderr
    if args.stderr:
        stderr_file = args.stderr
    else:
        stderr_file = 'stderr.log'  # 默认文件名

    # redirect stderr
    stderr_logger = tee.StderrTee(stderr_file, mode="w")
    loggers.append(stderr_logger)

# main program
if __name__ == '__main__':
    if args.train:
        with tee.StdoutTee(stdout_file), tee.StderrTee(stderr_file):
            env = mo_gym.make('rl-ctd-v0', current_design = 'mc_top', ReBuildRpt = True)
            # reset env and agent
            env.reset()
            agent = RL_CTD.create_agent(env)
            agent.train(
                total_timesteps=1,
                eval_env=env,
            )

# close all loggers
tee.close_all()
