# Many thanks for this code from
# https://gist.github.com/senbon/70adf5410950c0dc882b
#
import cPickle as pickle
import numpy as np
import os

import lasagne as nn

__all__ = [
    'load_model',
    'save_model',
]

MODEL_EXTENSION = 'pms'
UPDATE_EXTENSION = 'ups'


def load_model(model, updates, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    if updates is not None:
        filename1 = os.path.join('./', '%s.%s' % (filename, UPDATE_EXTENSION))
        with open(filename1, 'r') as f:
            data1 = pickle.load(f)
        for p, value in zip(updates.keys(), data1):
            p.set_value(value.astype(np.float32))

    filename2 = os.path.join('./', '%s.%s' % (filename, MODEL_EXTENSION))
    with open(filename2, 'r') as f:
        data2 = pickle.load(f)
    for i in range(len(data2)):
        data2[i] = data2[i].astype(np.float32)
    nn.layers.set_all_param_values(model, data2)


def save_model(model, updates, filename):
    """Pickels the parameters within a Lasagne model."""
    if updates is not None:
        data1 = [p.get_value() for p in updates.keys()]
        filename1 = os.path.join('./', filename)
        filename1 = '%s.%s' % (filename1, UPDATE_EXTENSION)
        with open(filename1, 'w') as f:
            pickle.dump(data1, f)

    data2 = nn.layers.get_all_param_values(model)
    filename2 = os.path.join('./', filename)
    filename2 = '%s.%s' % (filename2, MODEL_EXTENSION)
    with open(filename2, 'w') as f:
        pickle.dump(data2, f)
