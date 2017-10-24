import sys
import os


# device example: cpu, cuda0, cuda1, gpu0, gpu1
def set_environ(device, cnmem):
    device_env = "device=" + device
    theano_env = "mode=FAST_RUN," + device_env + ",floatX=float32,"

    if device != "cpu":
        cnmem_env = "lib.cnmem=" + cnmem
        theano_env = theano_env + "," + cnmem_env

    os.environ["THEANO_FLAGS"] = theano_env


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

    best_acc = 0.0
    worst_acc = 1.0

    for i in range(1):
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

        if all_acc / all_n > best_acc:
            best_acc = all_acc / all_n
        if all_acc / all_n < worst_acc:
            worst_acc = all_acc / all_n

        list_all_target.extend(all_target)
        list_all_output.extend(all_output)
        list_all_acc.append([all_acc / all_n])

    print best_acc
    print worst_acc

    np.save(("CAD_acc_" + modes[mode]), list_all_acc)
    np.save(("CAD_target_" + modes[mode]), list_all_target)
    np.save(("CAD_output_" + modes[mode]), list_all_output)


def cad():
    for i in [0, 4]:
        test_cad(mode=i)


# parameter : gpu device, cnmem value, dataset
# list datasets: cad, sbu, m2i
if __name__ == '__main__':
    set_environ(sys.argv[1], sys.argv[2])
    # set_environ("cpu", "0.8")
    from solver.cad_solver import *

    cad()

    if sys.argv[3] == 'cad':
        print "CAD"
        cad()
    elif sys.argv[3] == 'sbu':
        print "SBU"
    elif sys.argv[3] == 'm2i':
        print "M2I"
    else:
        print "Fail..."
