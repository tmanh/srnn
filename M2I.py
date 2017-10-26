import numpy as np
from utils import *


def get_permuted_data(data, fold, permutations):
    start = [0, 8, 16, 24, 32]
    end = [8, 16, 24, 32, 40]

    return data[permutations[start[fold]:end[fold]]]


def m2i_get_data(y, thh1, thh2, shh, ske1, ske2, fold, view, permutations):
    # 5 for test and 16 for train
    train_y = None
    train_thh1 = None
    train_thh2 = None
    train_ske1 = None
    train_ske2 = None
    train_shh = None

    test_y = None
    test_thh1 = None
    test_thh2 = None
    test_ske1 = None
    test_ske2 = None
    test_shh = None

    for i in range(len(y) / 2):
        k = 2 * i + view

        for f in range(5):
            if f % 5 != fold:
                if train_y is None:
                    train_y = get_permuted_data(y[k], f, permutations)
                    train_shh = get_permuted_data(shh[k], f, permutations)
                    train_thh1 = get_permuted_data(thh1[k], f, permutations)
                    train_thh2 = get_permuted_data(thh2[k], f, permutations)
                    train_ske1 = get_permuted_data(ske1[k], f, permutations)
                    train_ske2 = get_permuted_data(ske2[k], f, permutations)

                else:
                    train_y = np.concatenate((train_y, get_permuted_data(y[k], f, permutations)), axis=0)
                    train_shh = np.concatenate((train_shh, get_permuted_data(shh[k], f, permutations)), axis=0)
                    train_thh1 = np.concatenate((train_thh1, get_permuted_data(thh1[k], f, permutations)), axis=0)
                    train_thh2 = np.concatenate((train_thh2, get_permuted_data(thh2[k], f, permutations)), axis=0)
                    train_ske1 = np.concatenate((train_ske1, get_permuted_data(ske1[k], f, permutations)), axis=0)
                    train_ske2 = np.concatenate((train_ske2, get_permuted_data(ske2[k], f, permutations)), axis=0)
            else:
                if test_y is None:
                    test_y = get_permuted_data(y[k], f, permutations)
                    test_thh1 = get_permuted_data(thh1[k], f, permutations)
                    test_thh2 = get_permuted_data(thh2[k], f, permutations)
                    test_ske1 = get_permuted_data(ske1[k], f, permutations)
                    test_ske2 = get_permuted_data(ske2[k], f, permutations)
                    test_shh = get_permuted_data(shh[k], f, permutations)
                else:
                    test_y = np.concatenate((test_y, get_permuted_data(y[k], f, permutations)), axis=0)
                    test_thh1 = np.concatenate((test_thh1, get_permuted_data(thh1[k], f, permutations)), axis=0)
                    test_thh2 = np.concatenate((test_thh2, get_permuted_data(thh2[k], f, permutations)), axis=0)
                    test_ske1 = np.concatenate((test_ske1, get_permuted_data(ske1[k], f, permutations)), axis=0)
                    test_ske2 = np.concatenate((test_ske2, get_permuted_data(ske2[k], f, permutations)), axis=0)
                    test_shh = np.concatenate((test_shh, get_permuted_data(shh[k], f, permutations)), axis=0)

    return [train_y, train_thh1, train_thh2, train_ske1, train_ske2, train_shh,
            test_y, test_thh1, test_thh2, test_ske1, test_ske2, test_shh]
