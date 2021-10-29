import numpy as np
import tensorflow as tf
import warnings
from collections import defaultdict
import tf_util as U
import logger
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def sync_from_root(sess, variables, comm=None):
    """
    Send the root node's parameters to every worker.
    Arguments:
      sess: the TensorFlow session.
      variables: all parameter variables including optimizer's
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    values = comm.bcast(sess.run(variables))
    sess.run([tf.compat.v1.assign(var, val) for (var, val) in zip(variables, values)])


def mpi_weighted_mean(comm, local_name2valcount):
    """
    Perform a weighted average over dicts that are each on a different node
    Input: local_name2valcount: dict mapping key -> (value, count)
    Returns: key -> mean
    """
    all_name2valcount = comm.gather(local_name2valcount)
    if comm.rank == 0:
        name2sum = defaultdict(float)
        name2count = defaultdict(float)
        for n2vc in all_name2valcount:
            for (name, (val, count)) in n2vc.items():
                try:
                    val = float(val)
                except ValueError:
                    if comm.rank == 0:
                        warnings.warn('WARNING: tried to compute mean on non-float {}={}'.format(name, val))
                else:
                    name2sum[name] += val * count
                    name2count[name] += count
        return {name: name2sum[name] / name2count[name] for name in name2sum}
    else:
        return {}


class MpiAdam(object):
    def __init__(self, var_list, *, beta1=0.9, beta2=0.999, epsilon=1e-8, scale_grad_by_procs=True, comm=None):
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        size = sum(U.numel(v) for v in var_list)
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size, 'float32')
        self.t = 0
        self.setfromflat = U.SetFromFlat(var_list)
        self.getflat = U.GetFlat(var_list)
        self.comm = MPI.COMM_WORLD if comm is None and MPI is not None else comm

    def update(self, localg, stepsize):
        if self.t % 100 == 0:
            self.check_synced()
        localg = localg.astype('float32')
        if self.comm is not None:
            globalg = np.zeros_like(localg)
            self.comm.Allreduce(localg, globalg, op=MPI.SUM)
            if self.scale_grad_by_procs:
                globalg /= self.comm.Get_size()
        else:
            globalg = np.copy(localg)

        self.t += 1
        a = stepsize * np.sqrt(1 - self.beta2**self.t)/(1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = (-a) * self.m / (np.sqrt(self.v) + self.epsilon)
        self.setfromflat(self.getflat() + step)

    def sync(self):
        if self.comm is None:
            return
        theta = self.getflat()
        self.comm.Bcast(theta, root=0)
        self.setfromflat(theta)

    def check_synced(self):
        if self.comm is None:
            return
        if self.comm.Get_rank() == 0:
            theta = self.getflat()
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = self.getflat()
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)


class MpiAdamOptimizer(tf.compat.v1.train.AdamOptimizer):
    """Adam optimizer that averages gradients across mpi processes."""
    def __init__(self, comm, grad_clip=None, mpi_rank_weight=1, **kwargs):
        self.comm = comm
        self.grad_clip = grad_clip
        self.mpi_rank_weight = mpi_rank_weight
        tf.compat.v1.train.AdamOptimizer.__init__(self, **kwargs)

    def compute_gradients(self, loss, var_list, **kwargs):
        grads_and_vars = tf.compat.v1.train.AdamOptimizer.compute_gradients(self, loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = tf.concat([tf.reshape(g, (-1,)) for g, v in grads_and_vars], axis=0) * self.mpi_rank_weight
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        total_weight = np.zeros(1, np.float32)
        self.comm.Allreduce(np.array([self.mpi_rank_weight], dtype=np.float32), total_weight, op=MPI.SUM)
        total_weight = total_weight[0]

        buf = np.zeros(sum(sizes), np.float32)
        countholder = [0]  # Counts how many times _collect_grads has been called
        stat = tf.reduce_sum(grads_and_vars[0][1])  # sum of first variable

        def _collect_grads(flat_grad, np_stat):
            if self.grad_clip is not None:
                gradnorm = np.linalg.norm(flat_grad)
                if gradnorm > 1:
                    flat_grad /= gradnorm
                logger.logkv_mean('gradnorm', gradnorm)
                logger.logkv_mean('gradclipfrac', float(gradnorm > 1))
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(total_weight), out=buf)
            if countholder[0] % 100 == 0:
                check_synced(np_stat, self.comm)
            countholder[0] += 1
            return buf

        avg_flat_grad = tf.compat.v1.py_func(_collect_grads, [flat_grad, stat], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                              for g, (_, v) in zip(avg_grads, grads_and_vars)]
        return avg_grads_and_vars


def check_synced(localval, comm=None):
    """
    It's common to forget to initialize your variables to the same values, or
    (less commonly) if you update them in some other way than adam, to get them out of sync.
    This function checks that variables on all MPI workers are the same, and raises
    an AssertionError otherwise
    Arguments:
        comm: MPI communicator
        localval: list of local variables (list of variables on current worker to be compared with the other workers)
    """
    comm = comm or MPI.COMM_WORLD
    vals = comm.gather(localval)
    if comm.rank == 0:
        assert all(val == vals[0] for val in vals[1:]),\
            'MpiAdamOptimizer detected that different workers have different weights: {}'.format(vals)
