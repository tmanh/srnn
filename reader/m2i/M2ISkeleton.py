import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class M2ISkeleton(object):
    SKE_LENGTH = 60  # 45 for 15 joints xyz per 1 person

    def __init__(self):
        pass

    @staticmethod
    def read(filename):
        input_file = open(filename)

        lines = input_file.readlines()
        skeleton1 = np.zeros((len(lines), M2ISkeleton.SKE_LENGTH))
        skeleton2 = np.zeros((len(lines), M2ISkeleton.SKE_LENGTH))

        for i in range(len(lines)):
            elements = lines[i].split('\n')[0].split('\t')
            joint_list = [float(x) for x in elements[0:]]

            skeleton1[i, :] = np.asarray(joint_list[0:M2ISkeleton.SKE_LENGTH])
            skeleton2[i, :] = np.asarray(joint_list[M2ISkeleton.SKE_LENGTH:])

        input_file.close()

        return skeleton1, skeleton2

    @staticmethod
    def display_skeleton(ske):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        nx = range(0, 60, 3)
        ny = range(1, 60, 3)
        nz = range(2, 60, 3)

        ax.scatter(ske[nx[0]], ske[ny[0]], ske[nz[0]], c='r', marker='o')
        ax.scatter(ske[nx[1]], ske[ny[1]], ske[nz[1]], c='r', marker='o')
        ax.scatter(ske[nx[2]], ske[ny[2]], ske[nz[2]], c='r', marker='o')
        ax.scatter(ske[nx[3]], ske[ny[3]], ske[nz[3]], c='g', marker='o')
        ax.scatter(ske[nx[4]], ske[ny[4]], ske[nz[4]], c='g', marker='o')
        ax.scatter(ske[nx[5]], ske[ny[5]], ske[nz[5]], c='g', marker='o')
        ax.scatter(ske[nx[6]], ske[ny[6]], ske[nz[6]], c='c', marker='o')
        ax.scatter(ske[nx[7]], ske[ny[7]], ske[nz[7]], c='c', marker='o')
        ax.scatter(ske[nx[8]], ske[ny[8]], ske[nz[8]], c='c', marker='o')
        ax.scatter(ske[nx[9]], ske[ny[9]], ske[nz[9]], c='m', marker='o')
        ax.scatter(ske[nx[10]], ske[ny[10]], ske[nz[10]], c='m', marker='o')
        ax.scatter(ske[nx[11]], ske[ny[11]], ske[nz[11]], c='m', marker='o')
        ax.scatter(ske[nx[12]], ske[ny[12]], ske[nz[12]], c='y', marker='o')
        ax.scatter(ske[nx[13]], ske[ny[13]], ske[nz[13]], c='y', marker='o')
        ax.scatter(ske[nx[14]], ske[ny[14]], ske[nz[14]], c='y', marker='o')
        ax.scatter(ske[nx[15]], ske[ny[15]], ske[nz[15]], c='k', marker='o')
        ax.scatter(ske[nx[16]], ske[ny[16]], ske[nz[16]], c='k', marker='o')
        ax.scatter(ske[nx[17]], ske[ny[17]], ske[nz[17]], c='k', marker='o')
        ax.scatter(ske[nx[18]], ske[ny[18]], ske[nz[18]], c='k', marker='o')
        ax.scatter(ske[nx[19]], ske[ny[19]], ske[nz[19]], c='k', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()
