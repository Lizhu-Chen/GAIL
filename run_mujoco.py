import argparse
import os.path as osp
import logging
import numpy as np
import gym
from mpi4py import MPI
from tqdm import tqdm

import mlp_policy
import tf_util as U
from misc_util import set_global_seeds, boolean_flag
import logger
from bench.monitor import Monitor
from mujoco_dset import Mujoco_Dset
from adversary import TransitionClassifier


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='enviroment ID', default='Walker2d-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='data/deterministic.trpo.Walker2d.0.00.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)

    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')

    # for evaluation
    boolean_flag(parser, 'stochastic_policy', default=False, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectory or not')

    # Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=10)

    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)

    # Network Configuration (using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)

    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)

    # Training Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=30)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=500e3)  # iteration???????????????????????????timesteps_per_batch

    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=100)
    return parser.parse_args()


def get_task_name(args):
    task_name = args.algo
    if args.pretrained:
        task_name += "pretrained."
    task_name += '_' + args.env_id.split("-")[0]
    task_name += '.timesteps_' + str(args.num_timesteps) + ".g" + str(args.g_step) + ".d" + str(
                 args.d_step) + ".g_ent_" + str(args.policy_entcoeff) + ".d_ent_" + str(args.adversary_entcoeff)
    task_name += ".seed_" + str(args.seed)
    return task_name


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env_id)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
    env = Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)

    if args.task == 'train':
        dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
        reward_giver = TransitionClassifier(env, args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)
        train(env, args.seed, policy_fn, reward_giver, dataset, args.algo, args.g_step, args.d_step,
              args.policy_entcoeff, args.num_timesteps, args.save_per_iter, args.checkpoint_dir, args.log_dir,
              args.pretrained, args.BC_max_iter, task_name)
    elif args.task == 'evaluate':
        runner(env, policy_fn, args.load_model_path, timesteps_per_batch=1024, number_trajs=30,
               stochastic_policy=args.stochastic_policy, save=args.save_sample)
    else:
        raise NotImplementedError
    env.close()


def train(env, seed, policy_fn, reward_giver, dataset, algo, g_step, d_step, policy_entcoeff, num_timesteps,
          save_per_iter, checkpoint_dir, log_dir, pretrained, BC_max_iter, task_name=None):
    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        # pretrain with behavior cloning
        import behavior_clone
        pretrained_weight = behavior_clone.learn(env, policy_fn, dataset, max_iters=BC_max_iter)

    if algo == 'trpo':
        import trpo_mpi
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        trpo_mpi.learn(env, policy_fn, reward_giver, dataset, rank,
                       pretrained=pretrained, pretrained_weight=pretrained_weight,
                       g_step=g_step, d_step=d_step,
                       entcoeff=policy_entcoeff,
                       max_timesteps=num_timesteps,
                       ckpt_dir=checkpoint_dir, log_dir=log_dir,
                       save_per_iter=save_per_iter,
                       timesteps_per_batch=1024,
                       max_kl=0.01, cg_iters=10, cg_damping=0.1,
                       gamma=0.995, lam=0.97,
                       vf_iters=5, vf_stepsize=1e-3,
                       task_name=task_name)
    elif algo == 'ppo':
        if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
            rank = 0
            logger.configure_logger(args.log_path)
        else:
            rank = MPI.COMM_WORLD.Get_rank()
            logger.configure_logger(args.log_path, format_strs=[])
        print('Training {} on mujoco:{}'.format(args.algo, args.env_id))

        # load_path = './checkpoint/ppo_30e3_max1000/'\
        #             'ppo_gail.transition_limitation_-1.Hopper.timesteps_20000.0.seed_0.meta'
        # python -m run_mujoco --task=evaluate --load_model_path=checkpoint/ppo_gail.transition_limitation_-1.
        # Hopper.timesteps_200000.0.seed_0/ppo_gail.transition_limitation_-1.Hopper.timesteps_200000.0.seed_0

        import ppo2
        ppo2.learn(env=env, policy_func=policy_fn, reward_giver=reward_giver, expert_dataset=dataset, rank=rank,
                   pretrained=pretrained, pretrained_weight=pretrained_weight,
                   g_step=g_step, d_step=d_step, entcoeff=policy_entcoeff, cliprange=0.2,
                   max_timesteps=num_timesteps, ckpt_dir=checkpoint_dir, log_dir=log_dir,
                   save_per_iter=save_per_iter, timesteps_per_batch=1024, d_stepsize=3e-4,
                   gamma=0.99, lam=0.97, vf_iters=5, vf_stepsize=3e-4, task_name=task_name)

    else:
        raise NotImplementedError


def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    U.load_variables(load_model_path)

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list) / len(len_list)
    avg_ret = sum(ret_list) / len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    return avg_len, avg_ret


# Sample one trajectory (until trajectory end)
def traj_1_generator(pi, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []

    while True:
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)
        env.render()

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj


if __name__ == '__main__':
    args = argsparser()
    main(args)
