import time
import os
from contextlib import contextmanager
from mpi4py import MPI
from collections import deque

import tensorflow as tf
import numpy as np

import tf_util as U
from misc_util import zipsame, fmt_row
import dataset
import logger
from mpi_util import MpiAdam


def traj_segment_generator(pi, env, reward_giver, horizon, stochastic):  # pi是政策函数, horizon是timesteps_per_batch=1024
    # initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0  # reward from discriminator
    true_rew = 0.0  # reward from environment
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


def learn(env, policy_func, reward_giver, expert_dataset, rank,
          pretrained, pretrained_weight, *,
          g_step, d_step, entcoeff, save_per_iter,
          ckpt_dir, log_dir, timesteps_per_batch, task_name,
          gamma, lam, max_kl, cg_iters, cg_damping=1e-2,
          vf_stepsize=3e-4, d_stepsize=3e-4, vf_iters=3,
          max_timesteps, max_episodes=0, max_iters=0,
          callback=None
          ):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ---------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space  # Box(-1.0, 1.0, (3,), float32)
    pi = policy_func("pi", ob_space, ac_space, reuse=pretrained_weight != None)
    oldpi = policy_func("oldpi", ob_space, ac_space)
    atarg = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)  # D(pi)
    ent = pi.pd.entropy()  # entropy of policy, H(pi)
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent  # lambda*H(pi)

    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))  # error between value function and empirical return

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pinew / piold
    surrgain = tf.reduce_mean(ratio * atarg)  # surrogate advantage L = advantage * ratio
    optimgain = surrgain + entbonus

    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.startswith("pi/pol") or v.name.startswith("pi/logstd")]
    vf_var_list = [v for v in all_var_list if v.name.startswith("pi/vff")]
    assert len(var_list) == len(vf_var_list) + 1
    d_adam = MpiAdam(reward_giver.get_trainable_variables())
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([], [], updates=[tf.compat.v1.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(oldpi.get_variables(),
                                                                                pi.get_variables())])
    compute_vloss = U.function([ob, ret], vferr)
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)  # Fisher-Vector Product
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            # print(colorize(msg, color='magenta'))
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
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards
    true_rewbuffer = deque(maxlen=40)

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    # if provide pretrained weight
    if pretrained_weight is not None:
        U.load_variables(pretrained_weight, variables=pi.get_variables())

    g_loss, d_loss, ex_loss, g_acc, ex_acc, gss = [], [], [], [], [], []
    average_returns, true_rewards = [], []
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
            # saver.save(U.get_session(), fname)
            saver.save(tf.compat.v1.get_default_session(), fname)

        logger.log("********** Iteration %i ************" % (iters_so_far+1))

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p
        # ------------------ Update G ------------------
        meanvlosses, gs = [], []
        logger.log("Optimizing Policy...")
        for _ in range(g_step):
            with timed("sampling"):
                seg = seg_gen.__next__()  # generate trajectories
            add_vtarg_and_adv(seg, gamma, lam)
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            average_returns.append((sum(seg["ep_rets"])/len(seg["ep_rets"]))[0])
            true_rewards.append(sum(seg["ep_true_rets"])/len(seg["ep_true_rets"]))
            vpredbefore = seg["vpred"]  # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

            if hasattr(pi, "ob_rms"):
                pi.ob_rms.update(ob)  # update running mean/std for policy

            args = seg["ob"], seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]

            assign_old_eq_new()  # set old parameter values to new parameter values
            with timed("computegrad"):
                *lossbefore, g = compute_lossandgrad(*args)
            lossbefore = allmean(np.array(lossbefore))
            g = allmean(g)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
                from cg import cg
                with timed("cg"):
                    stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=(rank == 0))
                assert np.isfinite(stepdir).all()
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / max_kl)
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                surrbefore = lossbefore[0]
                stepsize = 1.0
                thbefore = get_flat()
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    set_from_flat(thnew)
                    meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                    improve = surr - surrbefore
                    # logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        logger.log("Got non-finite value of losses -- bad!")
                    elif kl > max_kl * 1.5:
                        logger.log("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        logger.log("surrogate didn't improve. shrinking step.")
                    else:
                        # logger.log("Stepsize OK!")
                        break
                    stepsize *= .5
                else:
                    logger.log("couldn't compute a good step")
                    set_from_flat(thbefore)
                if nworkers > 1 and iters_so_far % 20 == 0:
                    paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum()))  # list of tuples
                    assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
            with timed("vf"):
                for _ in range(vf_iters):
                    for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                             include_final_partial_batch=False, batch_size=128):
                        if hasattr(pi, "ob_rms"):
                            pi.ob_rms.update(mbob)  # update running mean/std for policy
                        g = allmean(compute_vflossandgrad(mbob, mbret))
                        vfadam.update(g, vf_stepsize)
        # print('meanvlosses:', len(meanvlosses))
        # g_losses = meanlosses
        # g_loss = surr + 0.5*v_loss 因为ppo用的是0.5
        # g_loss.append(surr+0.5*np.mean(meanvlosses))

        # for (lossname, lossval) in zip(loss_names, meanlosses):
        #     # if lossname == 'generator_acc':
        #     #     g_acc.append(lossval)
        #     logger.record_tabular(lossname, lossval)
        # logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        # logger.log(fmt_row(13, reward_giver.loss_name))
        ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob))
        batch_size = len(ob) // d_step
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for ob_batch, ac_batch in dataset.iterbatches((ob, ac),
                                                      include_final_partial_batch=False,
                                                      batch_size=batch_size):
            ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob_batch))
            # update running mean/std for reward_giver
            if hasattr(reward_giver, "obs_rms"):
                reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
            *newlosses, g = reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
            d_adam.update(allmean(g), d_stepsize)
            # print('d_g:', allmean(g))
            #d_losses.append(newlosses)
        # d_losses = np.mean(d_losses, axis=0)
        # logger.log(fmt_row(13, d_losses))
        # ex_acc.append(d_losses[-1])
        # g_acc.append(d_losses[-2])

        # lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"])  # local values
        # listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        # lens, rews, true_rets = map(flatten_lists, zip(*listoflrpairs))
        # true_rewbuffer.extend(true_rets)
        # lenbuffer.extend(lens)
        # rewbuffer.extend(rews)
        #
        # episodes_so_far += len(lens)
        timesteps_so_far += timesteps_per_batch
        # timesteps_so_far += sum(lens)
        iters_so_far += 1

        # if rank == 0:
        #     logger.dump_tabular()
        logger.log('cur_true_returns:', true_rewards[-1])
        logger.log('cur_average_returns:', average_returns[-1][0])

    from plot import loss_plot
    fname += 'true_rewards.txt'
    np.savetxt(fname, true_rewards, fmt='%.2f')
    loss_plot(true_rewards, average_returns, 'true_reward', 'average_returns')


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
