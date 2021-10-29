import time
import os
from contextlib import contextmanager
from mpi4py import MPI
from collections import deque

import tensorflow as tf
import numpy as np

import tf_util as U
from misc_util import zipsame
import dataset
import logger
from mpi_util import MpiAdam


def traj_segment_generator(pi, env, reward_giver, horizon, stochastic):  # pi是政策函数, horizon是timesteps_per_batch
    # initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0  # predicted reward from discriminator
    true_rew = 0.0  # true reward from environment
    ob = env.reset()

    cur_ep_ret = 0  # episode return = sum of rewards of all the steps of one episode
    cur_ep_len = 0
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []
    ep_lens = []

    # initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float32')
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')  # predicted value
    news = np.zeros(horizon, 'int32')  # done, True or False, if reached the goal
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()  # previous action

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news, "true_rew": true_rews,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "ep_true_rets": ep_true_rets}
            _, vpred = pi.act(stochastic, ob)
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        rew = reward_giver.get_reward(ob, ac)
        ob, true_rew, new, _ = env.step(ac)
        rews[i] = rew
        true_rews[i] = true_rew

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew
        cur_ep_len += 1
        if new:  # if new=True, one episode finished, reset every list
            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):  # generalized advantage estimation
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def constfn(val):
    def f(_):
        return val
    return f


def learn(*, env, policy_func, reward_giver, expert_dataset, rank,
          pretrained, pretrained_weight, lr=3e-4, cliprange,
          g_step, d_step, entcoeff, save_per_iter,
          ckpt_dir, log_dir, timesteps_per_batch, task_name,
          gamma, lam, d_stepsize, vf_iters, vf_stepsize,
          max_timesteps, max_episodes=0, max_iters=0, callback=None):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)

    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    # if isinstance(cliprange, float):
    #     cliprange = constfn(cliprange)
    # else:
    #     assert callable(cliprange)

    # Setup losses and stuff
    # ---------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space  # Box(-1.0, 1.0, (3,), float32)

    pi = policy_func("pi", ob_space, ac_space, reuse=pretrained_weight != None)
    oldpi = policy_func("oldpi", ob_space, ac_space)

    ob_ph = U.get_placeholder_cached(name="ob")
    ac_ph = pi.pdtype.sample_placeholder([None])

    ADV = tf.compat.v1.placeholder(tf.float32, [None])
    R = tf.compat.v1.placeholder(tf.float32, [None])
    OLDNEGLOGPAC = tf.compat.v1.placeholder(tf.float32, [None])

    # Calculate the entropy
    # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
    entropy = tf.reduce_mean(pi.pd.entropy())

    # CALCULATE THE LOSS
    # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

    # Clip the value to reduce variability during Critic training
    # Get the predicted value
    neglogpac = pi.pd.neglogp(ac_ph)
    vpred = pi.vpred
    vpredclipped = oldpi.vpred + tf.clip_by_value(pi.vpred - oldpi.vpred, -cliprange, cliprange)
    # Unclipped value
    vf_losses1 = tf.square(vpred - R)
    # Clipped value
    vf_losses2 = tf.square(vpredclipped - R)

    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

    # Calculate ratio (pi current policy / pi old policy)
    ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

    # Defining Loss = - J is equivalent to max J
    pg_losses = -ADV * ratio
    pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0-cliprange, 1.0+cliprange)

    # Final PG loss
    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
    clipfrac = tf.reduce_mean(tf.compat.v1.to_float(tf.greater(tf.abs(ratio - 1.0), cliprange)))

    # Total loss
    total_loss = pg_loss

    losses = [pg_loss, vf_loss, entropy, clipfrac]
    loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'clipfrac']

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
    vf_var_list = [v for v in all_var_list if v.name.startswith("pi/vff")]
    assert len(var_list) == len(vf_var_list) + 1
    d_adam = MpiAdam(reward_giver.get_trainable_variables())
    vfadam = MpiAdam(var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)

    assign_old_eq_new = U.function([], [], updates=[tf.compat.v1.assign(oldv, newv) for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob_ph, ac_ph, R, ADV, OLDNEGLOGPAC], total_loss)
    compute_lossandgrad = U.function([ob_ph, ac_ph, R, ADV, OLDNEGLOGPAC],
                                     losses + [U.flatgrad(total_loss, var_list)])
    compute_vflossandgrad = U.function([ob_ph, R], U.flatgrad(vf_loss, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            tstart = time.time()
            yield
            # print("done in %.3f seconds" % (time.time() - tstart))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    d_adam.sync()
    vfadam.sync()
    if rank == 0:
        print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, reward_giver, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 1
    tstart = time.time()
    lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards
    true_rewbuffer = deque(maxlen=40)

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    # if provide pretrained weight
    if pretrained_weight is not None:
        U.load_variables(pretrained_weight, variables=pi.get_variables())

    g_loss, d_loss = [], []
    average_returns, true_rewards, pred_rewards, max_returns, max_true_returns = [], [], [], [], []
    while True:
        if callback:
            callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        # Save model
        if rank == 0 and iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            fname = os.path.join(ckpt_dir, task_name)
            U.save_variables(fname)
            print("the save path is", fname)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            saver = tf.compat.v1.train.Saver()
            saver.save(tf.compat.v1.get_default_session(), fname)

            reward_name = fname + '_true_rewards_' + 'iters_' + str(iters_so_far) + 'of' + str(int(max_timesteps // timesteps_per_batch)) + '.txt'
            return_name = fname + '_average_returns_' + 'iters_' + str(iters_so_far) + 'of' + str(int(max_timesteps // timesteps_per_batch)) + '.txt'
            g_loss_name = fname + '_g_loss_' + 'iters_' + str(iters_so_far) + 'of' + str(int(max_timesteps // timesteps_per_batch)) + '.txt'
            d_loss_name = fname + '_d_loss_' + 'iters_' + str(iters_so_far) + 'of' + str(int(max_timesteps // timesteps_per_batch)) + '.txt'
            np.savetxt(reward_name, true_rewards, fmt='%.2f')
            np.savetxt(return_name, average_returns, fmt='%.2f')
            np.savetxt(g_loss_name, g_loss, fmt='%.2f')
            np.savetxt(d_loss_name, d_loss, fmt='%.2f')
            print('rewards are saved!')

        logger.log("********** Iteration %i ************" % iters_so_far)

        # ------------------ Update G ------------------
        for _ in range(g_step):
            logger.log("Optimizing Generator...")
            frac = 1.0 - (iters_so_far - 1.0) / timesteps_per_batch
            # Calculate the learning rate
            lrnow = lr(frac)

            with timed("sampling"):
                seg = seg_gen.__next__()  # generate trajectories
            add_vtarg_and_adv(seg, gamma, lam)
            ob, ac, atarg, tdlamret, news = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg["new"]
            average_returns.append((sum(seg["ep_rets"]) / len(seg["ep_rets"]))[0])
            true_rewards.append(sum(seg["ep_true_rets"]) / len(seg["ep_true_rets"]))
            vpredbefore = seg["vpred"]  # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

            # Calculate oldneglogpacs
            oldneglogpacs = oldpi.pd.neglogp(ac)
            feed_dict = {ob_ph: U.adjust_shape(ob_ph, ob)}
            U.get_session().run(oldneglogpacs, feed_dict)
            old_neglogpacs = tf.compat.v1.get_default_session().run(oldneglogpacs, feed_dict)
            old_neglogpacs = np.asarray(old_neglogpacs, dtype=np.float32)

            if hasattr(pi, "ob_rms"):
                pi.ob_rms.update(ob)  # update running mean/std for policy

            args = seg["ob"], seg["ac"], atarg, tdlamret, old_neglogpacs  # cliprangenow

            assign_old_eq_new()  # set old parameter values to new parameter values
            with timed("computegrad"):
                *lossbefore, g = compute_lossandgrad(*args)
            lossbefore = allmean(np.array(lossbefore))
            g = allmean(g)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")

            with timed("vf"):
                for _ in range(vf_iters):
                    for (mbob, mbac, mbatarg, mbret) in dataset.iterbatches((seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]),
                                                                            include_final_partial_batch=False, batch_size=128):

                        mbatarg = np.asarray((mbatarg - mbatarg.mean()) / mbatarg.std())

                        mboldneglogpacs = oldpi.pd.neglogp(mbac)
                        feed_dict = {ob_ph: U.adjust_shape(ob_ph, mbob)}
                        U.get_session().run(mboldneglogpacs, feed_dict)
                        mb_oldneglogpacs = tf.compat.v1.get_default_session().run(mboldneglogpacs, feed_dict)
                        mb_oldneglogpacs = np.asarray(mb_oldneglogpacs, dtype=np.float32)

                        if hasattr(pi, "ob_rms"):
                            pi.ob_rms.update(mbob)  # update running mean/std for policy

                        *lossbefore, g = compute_lossandgrad(mbob, mbac, mbatarg, mbret, mb_oldneglogpacs)
                        totalloss = compute_losses(mbob, mbac, mbatarg, mbret, mb_oldneglogpacs)
                        g_loss.append(totalloss)
                        g = allmean(g)
                        vfadam.update(g, vf_stepsize)

        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob))
        batch_size = len(ob) // d_step
        for ob_batch, ac_batch in dataset.iterbatches((ob, ac),
                                                      include_final_partial_batch=False,
                                                      batch_size=batch_size):
            ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob_batch))
            # update running mean/std for reward_giver
            if hasattr(reward_giver, "obs_rms"):
                reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
            *newlosses, g = reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
            d_total_loss = reward_giver.compute_total_loss(ob_batch, ac_batch, ob_expert, ac_expert)
            d_loss.append(d_total_loss)
            d_adam.update(allmean(g), d_stepsize)

        timesteps_so_far += timesteps_per_batch
        iters_so_far += 1

        # if rank == 0:
        #     logger.dump_tabular()

        logger.log('vf_stepsize:', vf_stepsize)
        logger.log('d_stepsize:', d_stepsize)
        logger.log('cur_true_returns:', true_rewards[-1])
        logger.log('cur_average_returns:', average_returns[-1][0])

    from plot import loss_plot
    reward_name = fname + '_true_rewards.txt'
    return_name = fname + '_average_returns.txt'
    np.savetxt(reward_name, true_rewards, fmt='%.2f')
    np.savetxt(return_name, average_returns, fmt='%.2f')
    np.savetxt(fname + '_g_loss.txt', g_loss, fmt='%.2f')
    np.savetxt(fname + '_d_loss.txt', d_loss, fmt='%.2f')
    loss_plot(true_rewards, average_returns, g_loss, 'true_reward', 'average_returns', 'generator_loss')


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
