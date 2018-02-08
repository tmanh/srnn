# Anh Minh Truong
import sys
import lasagne
import theano
import theano.tensor as T
import numpy as np

import lasagne.layers as llayers
import lasagne.objectives as lobjectives
import lasagne.regularization as lregularization

from lasagne.nonlinearities import leaky_rectify, softmax
from layers.sum_feature import *
from utils import *


REGULARIZE_RATE = 1e-4
# All gradients above this will be clipped
GRAD_CLIP = 100
# Optimization learning rate
LEARNING_RATE = 1e-2


class SBUSolver(object):
    def __init__(self, train_y, train_ske1, train_ske2, train_thh1,
                 train_thh2, train_shh, test_y, test_ske1, test_ske2,
                 test_thh1, test_thh2, test_shh, fold):
        self.input_var1 = None
        self.input_var2 = None
        self.input_var3 = None
        self.target_var = None

        self.l_out = None

        self.loss = None
        self.params = None
        self.updates = None
        self.prediction = None
        self.test_prediction = None

        self.train_fn = None
        self.test_fn = None

        self.trY = train_y
        self.trSKE1 = train_ske1
        self.trSKE2 = train_ske2
        self.trTHH1 = train_thh1
        self.trTHH2 = train_thh2
        self.trSHH = train_shh

        self.teY = test_y
        self.teSKE1 = test_ske1
        self.teSKE2 = test_ske2
        self.teTHH1 = test_thh1
        self.teTHH2 = test_thh2
        self.teSHH = test_shh

        self.mode = "null"
        self.fold = fold

        self.create_srnn()
        self.create_function()

    # noinspection PyTypeChecker
    def get_solver(self, mode=0):
        switcher = {
            0: self.srnn,
        }
        # Get the function from switcher dictionary
        solver = switcher.get(mode, lambda: "nothing")
        # Execute the function
        return solver

    def get_mode(self, mode=0):
        switcher = {
            0: "sbu.srnn",
        }

        self.mode = switcher.get(mode, "nothing")

    def create_srnn(self):
        # create Theano variables for input and target minibatch
        self.input_var1 = T.ftensor3('X1')
        self.input_var2 = T.ftensor3('X2')
        self.input_var3 = T.ftensor3('X3')
        self.target_var = T.ivector('Y')

        shape1 = (None, self.teTHH1.shape[1], self.trTHH1.shape[2])
        shape2 = (None, self.trTHH2.shape[1], self.trTHH2.shape[2])
        shape3 = (None, self.trSHH.shape[1], self.trSHH.shape[2])

        # create a small neural network
        l_thh1 = llayers.InputLayer(shape=shape1, input_var=input_var1)
        l_thh2 = llayers.InputLayer(shape=shape2, input_var=input_var2)
        l_shh = llayers.InputLayer(shape=shape3, input_var=input_var3)

        l_lthh1 = llayers.LSTMLayer(l_thh1, 128, grad_clipping=GRAD_CLIP)
        l_lthh2 = llayers.LSTMLayer(l_thh2, 128, grad_clipping=GRAD_CLIP)
        l_lshh = llayers.LSTMLayer(l_shh, 128, grad_clipping=GRAD_CLIP)

        l_concat = lasagne.layers.ConcatLayer([l_lthh1, l_lshh, l_lthh2], axis=2)
        l_drop = llayers.DropoutLayer(l_concat)

        l_hidden = llayers.LSTMLayer(l_drop, 384, grad_clipping=GRAD_CLIP, only_return_final=True)
        l_drop2 = llayers.DropoutLayer(l_hidden)

        self.l_out = llayers.DenseLayer(l_drop2, num_units=9, nonlinearity=lasagne.nonlinearities.softmax)

    @staticmethod
    def create_loss(prediction, loutput, target_var):
        rterm = REGULARIZE_RATE * lregularization.regularize_network_params(loutput, lregularization.l2)
        loss = lobjectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean() + rterm

        return loss

    @staticmethod
    def create_testfn(dim, prediction):
        return theano.function(dim, tensor.argmax(prediction, axis=1), allow_input_downcast=True)

    def create_function(self):
        # create loss function
        self.prediction = llayers.get_output(self.l_out)
        self.loss = SBUSolver.create_loss(self.prediction, self.l_out, self.target_var)

        # create parameter update expressions
        # create parameter update expressions
        self.params = llayers.get_all_params(self.l_out, trainable=True)
        self.updates = lasagne.updates.adagrad(self.loss, self.params, learning_rate=LEARNING_RATE)

        # compile training function that updates parameters
        # and returns training loss,
        # use trained network for predictions
        train_dim = [self.input_var1, self.input_var2, self.input_var3, self.target_var]
        test_dim = [self.input_var1, self.input_var2, self.input_var3]

        self.test_prediction = llayers.get_output(self.l_out, deterministic=True)

        self.train_fn = theano.function(train_dim, self.loss, updates=self.updates, allow_input_downcast=True)
        self.test_fn = SBUSolver.create_testfn(test_dim, self.test_prediction)

    def solve(self, mode=0):
        self.get_mode(mode)
        solver = self.get_solver(mode)
        best_acc, best_output, target = solver()
        return best_acc, best_output, target

    def srnn(self):
        best_acc = 0
        best_output = None
        target = self.teY.reshape((-1))

        # train network (assuming you've got some training data
        # in numpy arrays)
        for epoch in range(1501):
            if epoch % 10 == 0:
                predicted_output = self.test_fn(self.teTHH1, self.teTHH2, self.teSHH)

                print "{}\n{}\n\n".format(self.teY.reshape((-1)), predicted_output)
                print np.mean((self.teY.reshape((-1)) == predicted_output).astype(int))

                acc = np.mean((self.teY.reshape((-1)) == predicted_output).astype(int))

                if acc > best_acc:
                    best_acc = acc
                    best_output = predicted_output
            loss = 0
            loss += self.train_fn(self.trTHH1, self.trTHH2, self.trSHH, self.trY.reshape((-1)))
            print("Epoch %d: Loss %g" % (epoch + 1, loss))

        return best_acc, best_output, target
