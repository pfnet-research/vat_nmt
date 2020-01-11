# -*- coding: utf-8 -*-
import warnings

import six
from chainer import training
from chainer.backends import cuda
from chainer.dataset import convert
from chainer.training.updaters import MultiprocessParallelUpdater
from chainer.training.updaters.multiprocess_parallel_updater import gather_grads, _get_nccl_data_type, scatter_grads, gather_params
try:
    from cupy.cuda import nccl

    _available = True
except ImportError:
    _available = False


class MultiProcessParallelUpdaterMod(training.updaters.MultiprocessParallelUpdater):
    """
    For VaswaniAdam, there are some modifications.
    Use fixed eps.
    """

    def __init__(self, iterators, optimizer, converter=convert.concat_examples,
                 devices=None):
        if not MultiprocessParallelUpdater.available():
            raise Exception(
                'NCCL is not enabled. MultiprocessParallelUpdater '
                'requires NCCL.\n'
                'Please reinstall chainer after you install NCCL.\n'
                '(see https://github.com/chainer/chainer#installation).')

        assert len(iterators) == len(devices)
        for iterator in iterators[1:]:
            assert len(iterator.dataset) == len(iterators[0].dataset)

        # Correct optimizer parameters for new minibatch size
        optim = optimizer.__class__.__name__
        if optim in ('Adam', 'AdaGrad', 'RMSprop'):
            optimizer.eps *= len(devices)
            warnings.warn('optimizer.eps is changed to {} '
                          'by MultiprocessParallelUpdater for new batch size.'.
                          format(optimizer.eps))
        elif optim in ('VaswaniAdam'):
            pass
        elif optim in ('RMSpropGraves', 'AdaDelta'):
            optimizer.eps *= len(devices) ** 2  # not quite right for AdaDelta
            warnings.warn('optimizer.eps is changed to {} '
                          'by MultiprocessParallelUpdater for new batch size.'.
                          format(optimizer.eps))
        elif hasattr(optimizer, 'lr'):
            optimizer.lr /= len(devices)
            warnings.warn('optimizer.lr is changed to {} '
                          'by MultiprocessParallelUpdater for new batch size.'.
                          format(optimizer.lr))

        super(MultiprocessParallelUpdater, self).__init__(
            iterator=iterators[0],
            optimizer=optimizer,
            converter=converter
        )

        if isinstance(devices, dict):
            main = devices.pop('main')
            devices = list(six.itervalues(devices))
            devices = [main] + devices
        if devices is None or any(device is None for device in devices):
            raise ValueError('must specify GPU devices')

        self._master = optimizer.target
        self._devices = devices
        self._mpu_iterators = iterators
        self._initialized = False

        self._pipes = []
        self._workers = []
        self.comm = None


class MultiProcessParallelUpdaterMod_VAD(MultiProcessParallelUpdaterMod):

    def __init__(self, iterators, optimizer, converter=convert.concat_examples,
                 devices=None):
            super(MultiProcessParallelUpdaterMod_VAD, self).__init__(
                iterators=iterators,
                optimizer=optimizer,
                converter=converter,
                devices=devices
            )

    def update_core(self):
        self.setup_workers()

        self._send_message(('update', None))
        with cuda.Device(self._devices[0]):
            # For reducing memory
            self._master.cleargrads()

            optimizer = self.get_optimizer('main')
            batch = self.get_iterator('main').next()
            batch = self.converter(batch, self._devices[0])

            loss = self._calc_loss(self._master, batch, cleargrads_func=self._master.cleargrads)

            self._master.cleargrads()
            loss.backward()

            # NCCL: reduce grads
            null_stream = cuda.Stream.null
            if self.comm is not None:
                gg = gather_grads(self._master)
                nccl_data_type = _get_nccl_data_type(gg.dtype)
                self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size,
                                 nccl_data_type, nccl.NCCL_SUM,
                                 0, null_stream.ptr)
                scatter_grads(self._master, gg)
                del gg
            optimizer.update()
            if self.comm is not None:
                gp = gather_params(self._master)
                nccl_data_type = _get_nccl_data_type(gp.dtype)
                self.comm.bcast(gp.data.ptr, gp.size, nccl_data_type,
                                0, null_stream.ptr)


    def _calc_loss(self, model, in_arrays, cleargrads_func):
        if isinstance(in_arrays, tuple):
            return model(*in_arrays, cleargrads_func=cleargrads_func)
        elif isinstance(in_arrays, dict):
            return model(**in_arrays, cleargrads_func=cleargrads_func)
        else:
            return model(in_arrays, cleargrads_func=cleargrads_func)
