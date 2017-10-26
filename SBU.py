import numpy as np


def sbu_get_data(y, ske1, ske2, thh1, thh2, shh, permutations, fold):
    # 5 for test and 16 for train
    train_y = None
    train_thh1 = None
    train_thh2 = None
    train_shh = None
    train_ske1 = None
    train_ske2 = None

    test_y = None
    test_thh1 = None
    test_thh2 = None
    test_shh = None
    test_ske1 = None
    test_ske2 = None

    folds = [[0, 1, 2, 3],
             [4, 5, 6, 7],
             [8, 9, 10, 11],
             [12, 13, 14, 15],
             [16, 17, 18, 19, 20]]

    for k in range(len(folds)):
        if k != fold:
            for i in range(len(folds[k])):
                if train_y is None:
                    train_y = y[permutations[folds[k][i]]]
                    train_ske1 = ske1[permutations[folds[k][i]]]
                    train_ske2 = ske2[permutations[folds[k][i]]]
                    train_thh1 = thh1[permutations[folds[k][i]]]
                    train_thh2 = thh2[permutations[folds[k][i]]]
                    train_shh = shh[permutations[folds[k][i]]]
                else:
                    train_y = np.concatenate((train_y, y[permutations[folds[k][i]]]), axis=0)
                    train_ske1 = np.concatenate((train_ske1, ske1[permutations[folds[k][i]]]), axis=0)
                    train_ske2 = np.concatenate((train_ske2, ske2[permutations[folds[k][i]]]), axis=0)
                    train_thh1 = np.concatenate((train_thh1, thh1[permutations[folds[k][i]]]), axis=0)
                    train_thh2 = np.concatenate((train_thh2, thh2[permutations[folds[k][i]]]), axis=0)
                    train_shh = np.concatenate((train_shh, shh[permutations[folds[k][i]]]), axis=0)
        else:
            for i in range(len(folds[k])):
                if test_y is None:
                    test_y = y[permutations[folds[k][i]]]
                    test_ske1 = ske1[permutations[folds[k][i]]]
                    test_ske2 = ske2[permutations[folds[k][i]]]
                    test_thh1 = thh1[permutations[folds[k][i]]]
                    test_thh2 = thh2[permutations[folds[k][i]]]
                    test_shh = shh[permutations[folds[k][i]]]
                else:
                    test_y = np.concatenate((test_y, y[permutations[folds[k][i]]]), axis=0)
                    test_ske1 = np.concatenate((test_ske1, ske1[permutations[folds[k][i]]]), axis=0)
                    test_ske2 = np.concatenate((test_ske2, ske2[permutations[folds[k][i]]]), axis=0)
                    test_thh1 = np.concatenate((test_thh1, thh1[permutations[folds[k][i]]]), axis=0)
                    test_thh2 = np.concatenate((test_thh2, thh2[permutations[folds[k][i]]]), axis=0)
                    test_shh = np.concatenate((test_shh, shh[permutations[folds[k][i]]]), axis=0)

    return [train_y, train_ske1, train_ske2, train_thh1, train_thh2, train_shh,
            test_y, test_ske1, test_ske2, test_thh1, test_thh2, test_shh]
