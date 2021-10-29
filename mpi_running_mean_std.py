try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import tensorflow as tf
import numpy as np
import tf_util as U


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-2, shape=()):
        self._sum = tf.compat.v1.get_variable(dtype=tf.float64, shape=shape, initializer=tf.constant_initializer(0.0),
                                              name="runningsum", trainable=False)
        self._sumsq = tf.compat.v1.get_variable(dtype=tf.float64, shape=shape,
                                                initializer=tf.constant_initializer(epsilon), name="runningsumsq",
                                                trainable=False)
        self._count = tf.compat.v1.get_variable(dtype=tf.float64, shape=(),
                                                initializer=tf.constant_initializer(epsilon),
                                                name="count", trainable=False)
        self.shape = shape
        self.mean = tf.compat.v1.to_float(self._sum / self._count)
        self.std = tf.sqrt(tf.maximum(tf.compat.v1.to_float(self._sumsq / self._count) - tf.square(self.mean), 1e-2))

        newsum = tf.compat.v1.placeholder(shape=self.shape, dtype=tf.float64, name='sum')
        newsumsq = tf.compat.v1.placeholder(shape=self.shape, dtype=tf.float64, name='var')
        newcount = tf.compat.v1.placeholder(shape=[], dtype=tf.float64, name='count')

        self.incfiltparams = U.function([newsum, newsumsq, newcount], [],
                                        updates=[tf.compat.v1.assign_add(self._sum, newsum),
                                                 tf.compat.v1.assign_add(self._sumsq, newsumsq),
                                                 tf.compat.v1.assign_add(self._count, newcount)])

    def update(self, x):
        x = x.astype('float64')
        n = int(np.prod(self.shape))
        totalvec = np.zeros(n*2+1, 'float64')
        addvec = np.concatenate([x.sum(axis=0).ravel(), np.square(x).sum(axis=0).ravel(),
                                 np.array([len(x)], dtype='float64')])
        if MPI is not None:
            MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
        self.incfiltparams(totalvec[0:n].reshape(self.shape), totalvec[n:2*n].reshape(self.shape), totalvec[2*n])
