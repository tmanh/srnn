import theano.tensor as T
import numpy as np

from lasagne.layers.base import Layer
from lasagne.utils import as_tuple


__all__ = [
    "Softmax"
]


class Softmax(Layer):
    """
    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """
    def __init__(self, incoming, **kwargs):
        super(Softmax, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        X = input

        is_tensor3 = X.ndim > 2
        shape = X.shape

        if is_tensor3:
            X = X.reshape((shape[0] * shape[1], shape[2]))

        out = T.nnet.softmax(X)

        if is_tensor3:
            out = out.reshape((shape[0], shape[1], shape[2]))

        return out
