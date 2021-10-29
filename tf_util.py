import multiprocessing
import copy
import numpy as np
import tensorflow as tf
import os
import collections
from mpi4py import MPI


def switch(condition, then_expression, else_expression):
    x_shape = copy.copy(then_expression.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),  # tf.cast用于改变数据类型 这里改为bool类型
                lambda: then_expression,  # tf.cond类似于if...else 满足条件后执行第一个
                lambda: else_expression)  # lambda，不满足则执行第二个lambda
    x.set_shape(x_shape)
    return x


ALREADY_INITIALIZED = set()


def initialize():
    new_variables = set(tf.compat.v1.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.compat.v1.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer


def function(inputs, outputs, updates=None, givens=None):
    if isinstance(outputs, list):  # 判断outputs是否是list
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.key(), f(*args, **kwargs)))  # 变成和outputs一样的类型的数据
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]


class _Function(object):
    def __init__(self, inputs, outputs, updates, givens):
        for input in inputs:
            if not hasattr(input, 'make_feed_dict') and not (type(input) is tf.Tensor and len(input.op.inputs) == 0):
                assert False, "inputs should all be placeholders, constants, or have a make_feed_dict method"
        self.inputs = inputs
        self.input_names = {inp.name.split("/")[-1].split(":")[0]: inp for inp in inputs}
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens

    def _feed_input(self, feed_dict, inpt, value):
        if hasattr(inpt, 'make_feed_dict'):
            feed_dict.update(inpt.make_feed_dict(value))
        else:
            feed_dict[inpt] = adjust_shape(inpt, value)

    def __call__(self, *args, **kwargs):
        assert len(args) + len(kwargs) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        for inpt in self.givens:  # update feed dict with givens
            feed_dict[inpt] = adjust_shape(inpt, feed_dict.get(inpt, self.givens[inpt]))
        for inpt, value in zip(self.inputs, args):  # update the args
            self._feed_input(feed_dict, inpt, value)
        for inp_name, value in kwargs.items():
            self._feed_input(feed_dict, self.input_names[inp_name], value)
        results = get_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        return results


def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), "shape function assumes that shape is fully known"
    return out


def numel(x):
    return intprod(var_shape(x))


def intprod(x):
    return int(np.prod(x))


def flatgrad(loss, var_list, clip_norm=None):
    grads = tf.gradients(loss, var_list)
    if clip_norm is not None:
        grads = [tf.clip_by_norm(grad, clip_norm=clip_norm) for grad in grads]
    return tf.concat(axis=0, values=[tf.reshape(grad if grad is not None else tf.zeros_like(v), [numel(v)])
                                     for (v, grad) in zip(var_list, grads)])


class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])

        self.theta = theta = tf.compat.v1.placeholder(dtype, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = intprod(shape)
            assigns.append(tf.compat.v1.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        tf.compat.v1.get_default_session().run(self.op, feed_dict={self.theta: theta})


class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])

    def __call__(self):
        return tf.compat.v1.get_default_session().run(self.op)


def save_variables(save_path, variables=None, sess=None):
    import joblib
    sess = sess or get_session()
    variables = variables or tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)

    ps = sess.run(variables)
    save_dict = {v.name: value for v, value in zip(variables, ps)}
    dirname = os.path.dirname(save_path)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(save_dict, save_path)


def adjust_shape(placeholder, data):
    if not isinstance(data, np.ndarray) and not isinstance(data, list):
        return data
    if isinstance(data, list):
        data = np.array(data)
    placeholder_shape = [x or -1 for x in placeholder.shape.as_list()]
    assert _check_shape(placeholder_shape, data.shape), \
        'Shape of data {} is not compatible with shape of the paleceholder {}'.format(data.shape, placeholder_shape)
    return np.reshape(data, placeholder_shape)


def _check_shape(placeholder_shape, data_shape):
    # return True  # 为啥一进来就要返回 True
    squeezed_placeholder_shape = _squeeze_shape(placeholder_shape)
    squeezed_data_shape = _squeeze_shape(data_shape)
    for i, s_data in enumerate(squeezed_data_shape):
        s_placeholder = squeezed_placeholder_shape[i]
        if s_placeholder != -1 and s_data != s_placeholder:
            return False
    return True


def _squeeze_shape(shape):
    return [x for x in shape if x != 1]


def get_session(config=None):  # get default session or create one with a given config
    sess = tf.compat.v1.get_default_session()
    if sess is None:
        sess = make_session(config=config, make_default=True)
    return sess


def make_session(config=None, num_cpu=None, make_default=False, graph=None):
    # return a session that will use <num_cpu> CPU's only
    if num_cpu is None:
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
    if config is None:
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                inter_op_parallelism_threads=num_cpu,
                                intra_op_parallelism_threads=num_cpu)
        config.gpu_options.allow_growth = True
    if make_default:
        return tf.compat.v1.InteractiveSession(config=config, graph=graph)
    else:
        return tf.compat.v1.Session(config=config, graph=graph)


_PLACEHOLDER_CACHE = {}


def get_placeholder(name, dtype, shape):
    if name in _PLACEHOLDER_CACHE:
        out, dtype1, shape1 = _PLACEHOLDER_CACHE[name]
        if out.graph == tf.compat.v1.get_default_graph():
            assert dtype1 == dtype and shape1 == shape, \
                   'Placehoder with name {} has already been registered and has ' \
                   'shape {}, different from requested {}'.format(name, shape1, shape)
            return out
    out = tf.compat.v1.placeholder(dtype=dtype, shape=shape, name=name)
    _PLACEHOLDER_CACHE[name] = (out, dtype, shape)
    return out


def get_placeholder_cached(name):
    return _PLACEHOLDER_CACHE[name][0]


def load_state(fname, sess=None):
    import logger
    logger.warn('load_state method is deprecated, please use load_variables instead')
    sess = sess or get_session()
    saver = tf.compat.v1.train.Saver()
    saver.restore(tf.compat.v1.get_default_session(), fname)


def load_variables(load_path, variables=None, sess=None):
    import joblib
    sess = sess or get_session()
    variables = variables or tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)

    loaded_params = joblib.load(os.path.expanduser(load_path))
    # print('loaded_params:', loaded_params.keys())
    restores = []
    if isinstance(loaded_params, list):
        assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
        for d, v in zip(loaded_params, variables):
            restores.append(v.assign(d))
    else:
        # print('v:', variables)
        for v in variables:
            if v.name in loaded_params.keys():
                restores.append(v.assign(loaded_params[v.name]))

    sess.run(restores)


def dense(x, size, name, weight_init=None, bias_init=0, weight_loss_dict=None, reuse=None):
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        #print('name', tf.compat.v1.get_variable_scope().name)
        #assert (len(tf.compat.v1.get_variable_scope().name.split('/')) == 2)

        w = tf.compat.v1.get_variable("w", [x.get_shape()[1], size], initializer=weight_init)
        b = tf.compat.v1.get_variable("b", [size], initializer=tf.constant_initializer(bias_init))
        weight_decay_fc = 3e-4

        if weight_loss_dict is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(w), weight_decay_fc, name='weight_decay_loss')
            if weight_loss_dict is not None:
                weight_loss_dict[w] = weight_decay_fc
                weight_loss_dict[b] = 0.0

            tf.compat.v1.add_to_collection(tf.compat.v1.get_variable_scope().name.split('/')[0] + '_' + 'losses',
                                           weight_decay)

        return tf.nn.bias_add(tf.matmul(x, w), b)


def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.compat.v1.variable_scope(scope):
        nin = x.get_shape()[1]
        w = tf.compat.v1.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.compat.v1.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def save_state(fname, sess=None):
    import logger
    logger.warn('save_state method is deprecated, please use save_variables instead')
    sess = sess or get_session()
    dirname = os.path.dirname(fname)
    if any(dirname):
        os.makedirs(dirname, exist_ok=True)
    saver = tf.compat.v1.train.Saver()
    saver.save(tf.compat.v1.get_default_session(), fname)


def allmean(x):
    assert isinstance(x, np.ndarray)
    out = np.empty_like(x)
    nworkers = MPI.COMM_WORLD.Get_size()
    MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
    out /= nworkers
    return out
