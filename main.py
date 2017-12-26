import os
import sys
import cPickle


# device example: cpu, cuda0, cuda1, gpu0, gpu1
def set_environ(device, cnmem):
    device_env = "device=" + device
    theano_env = "mode=FAST_RUN," + device_env + ",floatX=float32,"

    if device != "cpu":
        cnmem_env = "lib.cnmem=" + cnmem
        theano_env = theano_env + "," + cnmem_env

    os.environ["THEANO_FLAGS"] = theano_env

#######################################################################################################################


def solve_cad(mode=0, fold="fold_1"):
    data_path = ("./data/CAD-120/" + fold)
    backup_path = "./backup/"

    solver = CADSolver(data_path, backup_path, fold)
    acc, n, target, output = solver.solve(mode)
    print acc, n
    return acc, n, target, output


def test_cad(mode=0):
    fold = ["fold_1", "fold_2", "fold_3", "fold_4"]
    modes = ["activity", "sub_detect", "obj_detect", "sub_anti", "obj_anti"]

    list_all_acc = []
    list_all_target = []
    list_all_output = []

    for i in range(5):
        all_acc = 0.0
        all_n = 0
        all_target = []
        all_output = []

        for f in fold:
            acc, n, target, output = solve_cad(mode=mode, fold=f)
            all_acc += acc
            all_n += n
            all_target.extend(target)
            all_output.extend(output)

            print all_acc / all_n

        list_all_target.extend(all_target)
        list_all_output.extend(all_output)
        list_all_acc.append([all_acc / all_n])

    np.save(("CAD_acc_" + modes[mode]), list_all_acc)
    np.save(("CAD_target_" + modes[mode]), list_all_target)
    np.save(("CAD_output_" + modes[mode]), list_all_output)

#######################################################################################################################


def m2i_evaluate(y, thh1, thh2, shh, ske1, ske2, view):
    list_acc = []
    list_output = []
    list_target = []

    all_acc = 0
    trials = 0

    for t in range(5):
        permutations = shuffle(40)

        for fold in range(5):
            trials += 1

            [train_y, train_thh1, train_thh2, train_ske1, train_ske2, train_shh,
             test_y, test_thh1, test_thh2, test_ske1, test_ske2, test_shh] = \
                m2i_get_data(y, thh1, thh2, shh, ske1, ske2, fold, view, permutations)

            solver = M2ISolver(train_y, train_thh1, train_thh2, train_ske1, train_ske2, train_shh,
                               test_y, test_thh1, test_thh2, test_ske1, test_ske2, test_shh)
            acc, output, target = solver.solve(view)

            all_acc += acc

            print (all_acc / trials)

            list_acc.append(acc)
            list_output.append(output)
            list_target.append(target)

    np.save("sub_m2i_acc.ny", list_acc)
    np.save("sub_m2i_output.ny", list_output)
    np.save("sub_m2i_target.ny", list_target)


def test_m2i(view):
    main_path = './data/M2I'
    path_to_dataset = '{1}/{0}.pik'.format("dataset", main_path)

    data = cPickle.load(open(path_to_dataset))

    y = data['labels']
    thh1 = data['temporal_human_human_1']
    thh2 = data['temporal_human_human_2']
    ske1 = data['ske_1']
    ske2 = data['ske_2']
    shh = data['spatial_human_human']

    m2i_evaluate(y, thh1, thh2, shh, ske1, ske2, view=view)

#######################################################################################################################


def sbu_evaluate(y, ske1, ske2, thh1, thh2, shh):
    list_acc = []
    list_output = []
    list_target = []

    all_acc = 0
    trials = 0

    for t in range(5):
        permutations = shuffle(len(list_y))

        for fold in range(5):
            trials += 1

            [train_y, train_ske1, train_ske2, train_thh1, train_thh2, train_shh,
             test_y, test_ske1, test_ske2, test_thh1, test_thh2, test_shh] = \
                get_train_test_data(y, ske1, ske2, thh1, thh2, shh, permutations, fold)

            solver = SBUSolver(train_y, train_ske1, train_ske2, train_thh1, train_thh2, train_shh,
                               test_y, test_ske1, test_ske2, test_thh1, test_thh2, test_shh, fold)
            acc, output, target = solver.solve()

            all_acc += acc

            print (all_acc / trials)

            list_acc.append(acc)
            list_output.append(output)
            list_target.append(target)

    np.save("sbu_acc.ny", list_acc)
    np.save("sbu_output.ny", list_output)
    np.save("sbu_target.ny", list_target)


def test_sbu():
    main_path = './data/SBU'
    path_to_dataset = '{1}/{0}.pik'.format("dataset", main_path)

    data = cPickle.load(open(path_to_dataset))

    y = data['labels']
    thh1 = data['temporal_human_human_1']
    thh2 = data['temporal_human_human_2']
    ske1 = data['ske_1']
    ske2 = data['ske_2']
    shh = data['spatial_human_human']

    sbu_evaluate(y, thh1, thh2, shh, ske1, ske2)

#######################################################################################################################


def cad():
    for i in range(0, 4):
        test_cad(mode=i)


def m2i():
    for i in range(0, 1):
        test_m2i(view=i)


def sbu():
    for i in range(0, 1):
        test_sbu()


# parameter : gpu device, cnmem value, dataset
# list datasets: cad, sbu, m2i
if __name__ == '__main__':
    set_environ(sys.argv[1], sys.argv[2])
    # set_environ("cpu", "0.8")

    if sys.argv[3] == 'cad':
        print "CAD"

        from solver.cad_solver import *
        cad()
    elif sys.argv[3] == 'sbu':
        print "SBU"

        from solver.sbu_solver import *
        sbu()
    elif sys.argv[3] == 'm2i':
        print "M2I"

        from solver.m2i_solver import *
        m2i()
    else:
        print "Fail..."
