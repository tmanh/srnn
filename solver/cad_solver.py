# Anh Minh Truong

import timeit
import cPickle
import lasagne
import theano
import theano.tensor as tensor
import numpy as np
import lasagne.layers as llayers
import lasagne.objectives as lobjectives
import lasagne.regularization as lregularization

# activation function
from lasagne.nonlinearities import leaky_rectify, softmax


# more layers
from layers.sum_feature import *
from layers.softmax import *

# utils
from utils import *

# All gradients above this will be clipped
GRAD_CLIP = 100
# Optimization learning rate
LEARNING_RATE = 0.1
# Regularization rate
REGULARIZE_RATE = 1e-5

MAXIMUM_TIME_STEP = 25
MAXIMUM_OBJECTS = 5


def permute(samples):
    x = np.random.permutation(samples)
    for i in range(5):
        x = x[np.random.permutation(samples)]
    return x


class CADSolver(object):
    def __init__(self, datapath, backuppath, fold):
        self.datapath = datapath
        self.backuppath = backuppath
        self.fold = fold

        self.input_var1 = tensor.ftensor3('X1')
        self.input_var2 = tensor.ftensor4('X2')
        self.input_var3 = tensor.ftensor4('X3')
        self.input_var4 = tensor.ftensor4('X4')
        self.target_var = tensor.ivector('y')

        self.test_activity = None
        self.test_human = None
        self.test_human_anticipation = None
        self.test_objects = None
        self.test_objects_anticipation = None
        self.test_thh = None
        self.test_soh = None
        self.test_too = None
        self.test_soo = None

        self.train_activity = None
        self.train_human = None
        self.train_human_anticipation = None
        self.train_objects = None
        self.train_objects_anticipation = None
        self.train_thh = None
        self.train_soh = None
        self.train_too = None
        self.train_soo = None

        self.mode = None

        self.l_actout = None
        self.l_subout = None
        self.l_objout = None

        self.aprediction = None
        self.sprediction = None
        self.oprediction = None

        self.aloss = None
        self.sloss = None
        self.oloss = None

        self.aparams = None
        self.sparams = None
        self.oparams = None

        self.aupdates = None
        self.supdates = None
        self.oupdates = None

        self.traino_fn = None
        self.traina_fn = None
        self.trains_fn = None

        self.testa_prediction = None
        self.tests_prediction = None
        self.testo_prediction = None

        self.predicta_fn = None
        self.predicts_fn = None
        self.predicto_fn = None

        self.get_data()
        self.create_srnn()
        self.create_function()

    def get_data(self):
        test_data = cPickle.load(
            open('{0}/test_data.pik'.format(self.datapath)))
        self.test_activity = test_data['labels_activity']
        self.test_human = test_data['labels_human']
        self.test_human_anticipation = test_data['labels_human_anticipation']
        self.test_objects = test_data['labels_objects']
        self.test_objects_anticipation = test_data['labels_objects_anticipation']
        self.test_thh = test_data['thh_features']
        self.test_soh = test_data['soh_features']
        self.test_too = test_data['too_features']
        self.test_soo = test_data['soo_features']

        train_data = cPickle.load(
            open('{0}/train_data.pik'.format(self.datapath)))
        self.train_activity = train_data['labels_activity']
        self.train_human = train_data['labels_human']
        self.train_human_anticipation = train_data['labels_human_anticipation']
        self.train_objects = train_data['labels_objects']
        self.train_objects_anticipation = train_data['labels_objects_anticipation']
        self.train_thh = train_data['thh_features']
        self.train_soh = train_data['soh_features']
        self.train_too = train_data['too_features']
        self.train_soo = train_data['soo_features']

    def get_cad_solver(self, mode):
        switcher = {
            0: self.activity,          # activity
            1: self.sub_detection,     # sub-activity (detect)
            2: self.obj_detection,     # object (detect)
            3: self.sub_anticipation,  # sub-activity (anti)
            4: self.obj_anticipation,  # object (anti)
        }
        # Get the function from switcher dictionary
        solver = switcher.get(mode, lambda: "nothing")
        # Execute the function
        return solver

    def set_mode_name(self, mode):
        switcher = {
            1: "cad.activity",
            0: "cad.sub_detection",
            2: "cad.obj_detection",
            3: "cad.sub_anticipation",
            4: "cad.obj_anticipation",
        }
        # Get the function from switcher dictionary
        self.mode = switcher.get(mode, "nothing")

    def solve(self, mode):
        self.set_mode_name(mode)

        solver = self.get_cad_solver(mode)
        acc, n, target, output = solver()

        return acc, n, target, output

    def create_srnn(self):
        # create Theano variables for input and target minibatch
        shape1 = (None, self.train_thh.shape[1], self.train_thh.shape[2])
        shape2 = (None, self.train_too.shape[1], self.train_too.shape[2], self.train_too.shape[3])
        shape3 = (None, self.train_soo.shape[1], self.train_soo.shape[2], self.train_soo.shape[3])
        shape4 = (None, self.train_soh.shape[1], self.train_soh.shape[2], self.train_soh.shape[3])

        # -------------------
        # Create input layers
        # -------------------
        l_thh = llayers.InputLayer(shape=shape1, input_var=self.input_var1)
        l_too = llayers.InputLayer(shape=shape2, input_var=self.input_var2)
        l_soo = llayers.InputLayer(shape=shape3, input_var=self.input_var3)
        l_soh = llayers.InputLayer(shape=shape4, input_var=self.input_var4)

        # -------------------
        # Create the first LSTM layer
        # -------------------
        l_lthh = llayers.LSTMLayer(l_thh, 128, grad_clipping=GRAD_CLIP)

        l_ltoo_reshape = llayers.ReshapeLayer(l_too, (-1, shape2[2], shape2[3]))
        l_lsoo_reshape = llayers.ReshapeLayer(l_soo, (-1, shape3[2], shape3[3]))
        l_lsoh_reshape = llayers.ReshapeLayer(l_soh, (-1, shape4[2], shape4[3]))

        l_ltoo = llayers.LSTMLayer(l_ltoo_reshape, 128, grad_clipping=GRAD_CLIP)
        l_lsoo = llayers.LSTMLayer(l_lsoo_reshape, 128, grad_clipping=GRAD_CLIP)
        l_lsoh = llayers.LSTMLayer(l_lsoh_reshape, 128, grad_clipping=GRAD_CLIP)

        l_dthh = llayers.DropoutLayer(l_lthh)

        # -------------------
        # Create fusion layer
        # -------------------
        l_oconcat = llayers.ConcatLayer([l_ltoo, l_lsoo, l_lsoh], axis=2)
        l_dobj = llayers.DropoutLayer(l_oconcat)

        # -------------------
        # Create the merge layer
        # -------------------
        l_oreshapeback = lasagne.layers.ReshapeLayer(l_dobj, (-1, shape3[1], shape3[2], 384))
        l_omerge = SumFeatureLayer(l_oreshapeback)

        # -------------------
        # Create the fusion layer
        # -------------------
        l_hconcat = llayers.ConcatLayer([l_dthh, l_omerge], axis=-1)

        # -------------------
        # Create the second LSTM layer
        # -------------------
        l_lact = llayers.LSTMLayer(l_hconcat, 256, grad_clipping=GRAD_CLIP, only_return_final=True)
        l_lsub = llayers.LSTMLayer(l_hconcat, 256, grad_clipping=GRAD_CLIP)
        l_lobj = llayers.LSTMLayer(l_dobj, 256, grad_clipping=GRAD_CLIP)

        l_dact = llayers.DropoutLayer(l_lact)
        l_dsub = llayers.DropoutLayer(l_lsub)
        l_dobj = llayers.DropoutLayer(l_lobj)

        # -------------------
        # Create the output layer
        # -------------------
        l_sub_reshape = lasagne.layers.ReshapeLayer(l_dsub, (-1, 256))
        l_obj_reshape = lasagne.layers.ReshapeLayer(l_dobj, (-1, 256))

        self.l_actout = llayers.DenseLayer(l_dact, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
        self.l_subout = llayers.DenseLayer(l_sub_reshape, num_units=11, nonlinearity=lasagne.nonlinearities.softmax)
        self.l_objout = llayers.DenseLayer(l_obj_reshape, num_units=13, nonlinearity=lasagne.nonlinearities.softmax)

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
        self.aprediction = lasagne.layers.get_output(self.l_actout)
        self.sprediction = lasagne.layers.get_output(self.l_subout)
        self.oprediction = lasagne.layers.get_output(self.l_objout)

        self.aloss = CADSolver.create_loss(self.aprediction, self.l_actout, self.target_var)
        self.sloss = CADSolver.create_loss(self.sprediction, self.l_subout, self.target_var)
        self.oloss = CADSolver.create_loss(self.oprediction, self.l_objout, self.target_var)

        # create parameter update expressions
        self.aparams = lasagne.layers.get_all_params(self.l_actout, trainable=True)
        self.sparams = lasagne.layers.get_all_params(self.l_subout, trainable=True)
        self.oparams = lasagne.layers.get_all_params(self.l_objout, trainable=True)

        self.aupdates = lasagne.updates.adagrad(self.aloss, self.aparams, learning_rate=LEARNING_RATE)
        self.supdates = lasagne.updates.adagrad(self.sloss, self.sparams, learning_rate=LEARNING_RATE)
        self.oupdates = lasagne.updates.adagrad(self.oloss, self.oparams, learning_rate=LEARNING_RATE)

        # compile training function that updates parameters
        # and returns training loss
        train_objdim = [self.input_var2, self.input_var3, self.input_var4, self.target_var]
        train_actdim = [self.input_var1, self.input_var2, self.input_var3, self.input_var4, self.target_var]
        train_subdim = [self.input_var1, self.input_var2, self.input_var3, self.input_var4, self.target_var]

        self.traina_fn = theano.function(train_actdim, self.aloss, updates=self.aupdates, allow_input_downcast=True)
        self.trains_fn = theano.function(train_subdim, self.sloss, updates=self.supdates, allow_input_downcast=True)
        self.traino_fn = theano.function(train_objdim, self.oloss, updates=self.oupdates, allow_input_downcast=True)

        # use trained network for predictions
        test_objdim = [self.input_var2, self.input_var3, self.input_var4]
        test_actdim = [self.input_var1, self.input_var2, self.input_var3, self.input_var4]
        test_subdim = [self.input_var1, self.input_var2, self.input_var3, self.input_var4]

        self.testa_prediction = llayers.get_output(self.l_actout, deterministic=True)
        self.tests_prediction = llayers.get_output(self.l_subout, deterministic=True)
        self.testo_prediction = llayers.get_output(self.l_objout, deterministic=True)

        self.predicta_fn = CADSolver.create_testfn(test_actdim, self.testa_prediction)
        self.predicts_fn = CADSolver.create_testfn(test_subdim, self.tests_prediction)
        self.predicto_fn = CADSolver.create_testfn(test_objdim, self.testo_prediction)

    def activity(self):
        # train network (assuming you've got some training data
        # in numpy arrays)
        f = open(self.mode + ".txt", 'w')
        best_acc = 0.0
        samples = 0

        best_target = []
        best_output = []
        for epoch in range(1501):
            if epoch % 10 == 0:
                n = self.test_activity.shape[0]
                target = []
                output = []

                predicted_output = self.predicta_fn(self.test_thh, self.test_too, self.test_soo, self.test_soh)
                acc = np.mean((predicted_output == self.test_activity).astype(int))

                # print "{}\n{}\n----".format(self.test_activity, predicted_output)
                f.write("{}\n{}\n----".format(self.test_activity, predicted_output))

                for i in range(self.test_activity.shape[0]):
                    target.append(self.test_activity[i])
                    output.append(predicted_output[i])

                print acc
                f.write(str(acc) + "\n")

                if samples == 0:
                    samples = n
                if best_acc < acc:
                    save_model(self.l_actout, self.aupdates, (self.backuppath + "srnn_acti_l_act" + self.fold))
                    save_model(self.l_objout, self.oupdates, (self.backuppath + "srnn_acti_l_obj" + self.fold))
                    save_model(self.l_subout, self.supdates, (self.backuppath + "srnn_acti_l_sub" + self.fold))
                    best_acc = acc
                    best_target = target
                    best_output = output

            start_train = timeit.default_timer()
            self.traina_fn(self.train_thh, self.train_too, self.train_soo, self.train_soh, self.train_activity)
            self.trains_fn(self.train_thh, self.train_too, self.train_soo, self.train_soh,
                           self.train_human.reshape((-1)))
            self.traino_fn(self.train_too, self.train_soo, self.train_soh, self.train_objects.reshape((-1)))
            stop_train = timeit.default_timer()
            print "Train time = " + str(stop_train - start_train) + " seconds"
            f.write("Train time = " + str(stop_train - start_train) + " seconds")

        f.close()

        print best_acc
        return best_acc, samples, best_target, best_output

    def sub_detection(self):
        # train network (assuming you've got some training data
        # in numpy arrays)
        f = open(self.mode + ".txt", 'w')
        best_acc = 0.0
        samples = 0
        best_target = []
        best_output = []
        for epoch in range(1501):
            if epoch % 10 == 0:
                acc = 0.0
                n = 0
                target = []
                output = []

                predicted_output = self.predicts_fn(self.test_thh, self.test_too, self.test_soo, self.test_soh)
                predicted_output = np.reshape(predicted_output, (-1, MAXIMUM_TIME_STEP))

                # print "{}\n{}\n----".format(self.test_human, predicted_output)
                f.write("{}\n{}\n----".format(self.test_human, predicted_output))

                for i in range(self.test_human.shape[0]):
                    for s in range(self.test_human.shape[1] - 1, -1, -1):
                        if self.test_human[i, s] == 0:
                            break

                        n += 1
                        if self.test_human[i, s] == predicted_output[i, s]:
                            acc += 1

                        target.append(self.test_human[i, s])
                        output.append(predicted_output[i, s])

                print acc / n
                f.write(str(acc / n) + "\n")

                if samples == 0:
                    samples = n
                if best_acc < acc:
                    save_model(self.l_objout, self.oupdates, (self.backuppath + "srnn_subdetect_l_obj" + self.fold))
                    save_model(self.l_subout, self.supdates, (self.backuppath + "srnn_subdetect_l_sub" + self.fold))
                    best_acc = acc
                    best_target = target
                    best_output = output

            start_train = timeit.default_timer()
            self.trains_fn(self.train_thh, self.train_too, self.train_soo, self.train_soh,
                           self.train_human.reshape((-1)))
            self.traino_fn(self.train_too, self.train_soo, self.train_soh, self.train_objects.reshape((-1)))
            stop_train = timeit.default_timer()
            print "Train time = " + str(stop_train - start_train) + " seconds"
            f.write("Train time = " + str(stop_train - start_train) + " seconds")

        f.close()

        print best_acc / samples
        return best_acc, samples, best_target, best_output

    # maximum object
    def obj_detection(self):
        # train network (assuming you've got some training data
        # in numpy arrays)
        f = open(self.mode + ".txt", 'w')
        best_acc = 0.0
        samples = 0
        best_target = []
        best_output = []
        for epoch in range(1501):
            if epoch % 10 == 0:
                acc = 0.0
                n = 0
                target = []
                output = []

                predicted_output = self.predicto_fn(self.test_too, self.test_soo, self.test_soh)
                predicted_output = np.reshape(predicted_output, (-1, MAXIMUM_OBJECTS, MAXIMUM_TIME_STEP))

                # print "{}\n{}\n----".format(self.test_objects, predicted_output)
                f.write("{}\n{}\n----".format(self.test_objects, predicted_output))

                for i in range(self.test_objects.shape[0]):
                    for o in range(self.test_objects.shape[1]):
                        for s in range(self.test_objects.shape[2] - 1, -1, -1):
                            if self.test_objects[i, o, s] == 0:
                                break

                            n += 1
                            if self.test_objects[i, o, s] == predicted_output[i, o, s]:
                                acc += 1

                            target.append(self.test_objects[i, o, s])
                            output.append(predicted_output[i, o, s])

                print acc / n
                f.write(str(acc / n) + "\n")

                if samples == 0:
                    samples = n
                if best_acc < acc:
                    save_model(self.l_objout, self.oupdates, (self.backuppath + "srnn_objdetect_l_obj" + self.fold))
                    save_model(self.l_subout, self.supdates, (self.backuppath + "srnn_objdetect_l_sub" + self.fold))
                    best_acc = acc
                    best_target = target
                    best_output = output

            start_train = timeit.default_timer()
            self.trains_fn(self.train_thh, self.train_too, self.train_soo, self.train_soh,
                           self.train_human.reshape((-1)))
            self.traino_fn(self.train_too, self.train_soo, self.train_soh, self.train_objects.reshape((-1)))
            stop_train = timeit.default_timer()
            print "Train time = " + str(stop_train - start_train) + " seconds"
            f.write("Train time = " + str(stop_train - start_train) + " seconds")

        f.close()

        return best_acc, samples, best_target, best_output

    def sub_anticipation(self):
        # train network (assuming you've got some training data
        # in numpy arrays)
        f = open(self.mode + ".txt", 'w')
        best_acc = 0.0
        samples = 0
        best_target = []
        best_output = []
        for epoch in range(1501):
            if epoch % 10 == 0:
                acc = 0.0
                n = 0
                target = []
                output = []

                predicted_output = self.predicts_fn(self.test_thh, self.test_too, self.test_soo, self.test_soh)
                predicted_output = np.reshape(predicted_output, (-1, MAXIMUM_TIME_STEP))

                # print "{}\n{}\n----".format(self.test_human_anticipation, predicted_output)
                f.write("{}\n{}\n----".format(self.test_human_anticipation, predicted_output))

                for i in range(self.test_human_anticipation.shape[0]):
                    for s in range(self.test_human_anticipation.shape[1] - 1, -1, -1):
                        if self.test_human_anticipation[i, s] == 0:
                            break

                        n += 1
                        if self.test_human_anticipation[i, s] == predicted_output[i, s]:
                            acc += 1

                        target.append(self.test_human_anticipation[i, s])
                        output.append(predicted_output[i, s])

                print acc / n
                f.write(str(acc / n) + "\n")

                if samples == 0:
                    samples = n
                if best_acc < acc:
                    save_model(self.l_objout, self.oupdates, (self.backuppath + "srnn_antisub_l_obj" + self.fold))
                    save_model(self.l_subout, self.supdates, (self.backuppath + "srnn_antisub_l_sub" + self.fold))
                    best_acc = acc
                    best_target = target
                    best_output = output

            start_train = timeit.default_timer()
            self.trains_fn(self.train_thh, self.train_too, self.train_soo, self.train_soh,
                           self.train_human_anticipation.reshape((-1)))
            self.traino_fn(self.train_too, self.train_soo, self.train_soh,
                           self.train_objects_anticipation.reshape((-1)))
            stop_train = timeit.default_timer()
            print "Train time = " + str(stop_train - start_train) + " seconds"
            f.write("Train time = " + str(stop_train - start_train) + " seconds")

        f.close()

        print best_acc / samples
        return best_acc, samples, best_target, best_output

    def obj_anticipation(self):
        # train network (assuming you've got some training data
        # in numpy arrays)
        f = open(self.mode + ".txt", 'w')
        best_acc = 0.0
        samples = 0
        best_target = []
        best_output = []
        for epoch in range(1501):
            if epoch % 10 == 0:
                acc = 0.0
                n = 0
                target = []
                output = []

                predicted_output = self.predicto_fn(self.test_too, self.test_soo, self.test_soh)
                predicted_output = np.reshape(predicted_output, (-1, MAXIMUM_OBJECTS, MAXIMUM_TIME_STEP))

                # print "{}\n{}\n----".format(self.test_objects_anticipation, predicted_output)
                f.write("{}\n{}\n----".format(self.test_objects_anticipation, predicted_output))

                for i in range(self.test_objects_anticipation.shape[0]):
                    for o in range(self.test_objects_anticipation.shape[1]):
                        for s in range(self.test_objects_anticipation.shape[2] - 1, -1, -1):
                            if self.test_objects_anticipation[i, o, s] == 0:
                                break

                            n += 1
                            if self.test_objects_anticipation[i, o, s] == predicted_output[i, o, s]:
                                acc += 1

                            target.append(self.test_objects_anticipation[i, o, s])
                            output.append(predicted_output[i, o, s])

                print acc / n
                f.write(str(acc / n) + "\n")

                if samples == 0:
                    samples = n
                if best_acc < acc:
                    save_model(self.l_objout, self.oupdates, (self.backuppath + "srnn_antiobj_l_obj" + self.fold))
                    save_model(self.l_subout, self.supdates, (self.backuppath + "srnn_antiobj_l_sub" + self.fold))
                    best_acc = acc
                    best_target = target
                    best_output = output

            start_train = timeit.default_timer()
            self.trains_fn(self.train_thh, self.train_too, self.train_soo, self.train_soh,
                           self.train_human.reshape((-1)))
            self.traino_fn(self.train_too, self.train_soo, self.train_soh,
                           self.test_objects_anticipation.reshape((-1)))
            stop_train = timeit.default_timer()
            print "Train time = " + str(stop_train - start_train) + " seconds"
            f.write("Train time = " + str(stop_train - start_train) + " seconds")

        f.close()

        return best_acc, samples, best_target, best_output
