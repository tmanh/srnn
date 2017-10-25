import numpy as np


class SBUSkeleton(object):
    SKE_LENGTH = 45  # 45 for 15 joints xyz per 1 person

    def __init__(self):
        pass

    @staticmethod
    def read(filename):
        input_file = open(filename)

        lines = input_file.readlines()
        skeleton1 = np.zeros((len(lines), SBUSkeleton.SKE_LENGTH))
        skeleton2 = np.zeros((len(lines), SBUSkeleton.SKE_LENGTH))

        for i in range(len(lines)):
            elements = lines[i].split(',')
            joint_list = [float(x) for x in elements[1:]]
            skeleton1[i, :] = np.asarray(joint_list[0:SBUSkeleton.SKE_LENGTH])
            skeleton2[i, :] = np.asarray(joint_list[SBUSkeleton.SKE_LENGTH:])

        input_file.close()

        return skeleton1, skeleton2
