import theano.tensor as tensor
import numpy as np
import lasagne

from lasagne.nonlinearities import leaky_rectify, softmax
from lasagne.layers.base import Layer
from lasagne.layers.base import MergeLayer
from lasagne.utils import as_tuple

__all__ = [
    "SumFeatureLayer",
    "Softmax",
]


class SumFeatureLayer(Layer):
    """
    Sum Feature Layer
    Performs 1D sum over the trailing axis of a 3D input tensor to merge the
    output from recurrent neural network into a fixed size matrix.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, **kwargs):
        super(SumFeatureLayer, self).__init__(incoming, **kwargs)

        if len(self.input_shape) < 3 or len(self.input_shape) > 4:
            raise ValueError("Tried to create a temporal sum layer with "
                             "input shape %r. Expected 3 or 4 input "
                             "dimensions (batchsize, channels, 1 "
                             "spatial dimensions)."
                             % (self.input_shape,))

    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)  # copy / convert to mutable list
        if len(shape) == 3:
            return shape[0], shape[-1]
        else:
            return shape[0], shape[2], shape[-1]

    def get_output_for(self, input, **kwargs):
        return tensor.sum(input, axis=1)


class Softmax(Layer):
    def __init__(self, incoming, **kwargs):
        super(Softmax, self).__init__(incoming, **kwargs)

        if len(self.input_shape) < 2 or len(self.input_shape) > 2:
            raise ValueError("Tried to create a temporal sum layer with "
                             "input shape %r. Expected 2 input "
                             "dimensions (batchsize, features)."
                             % (self.input_shape,))

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return lasagne.nonlinearities.softmax(input)
