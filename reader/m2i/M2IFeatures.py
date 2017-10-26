import numpy as np
import math
import itertools


class M2IFeatures(object):
    def __init__(self):
        pass

    # ske1: frame t ~ ske2: frame t+1
    @staticmethod
    def joint_displacement(ske1, ske2):
        return ske2 - ske1

    # ske1: person 1 ~ ske2: person 2 ==> joint distance
    # ske1: person 1 frame t ~ ske2: person 1 frame t+1 ==> joint motion
    @staticmethod
    def joint_distance(ske1, ske2):
        n = ske1.shape[0]

        nx = range(0, n, 3)
        ny = range(1, n, 3)
        nz = range(2, n, 3)

        n_joints = len(nx)

        features = np.zeros((1, n_joints*n_joints))
        for i in range(len(nx)):
            for j in range(len(nx)):
                features[0, i*n_joints + j] = \
                    M2IFeatures.distance(
                        ske1[nx[i]], ske1[ny[i]], ske1[nz[i]],
                        ske2[nx[j]], ske2[ny[j]], ske2[nz[j]])

        return features

    @staticmethod
    def plane(ske1, ske2):
        n = ske1.shape[0]

        nx = range(0, n, 3)
        ny = range(1, n, 3)
        nz = range(2, n, 3)

        njoints = len(nx)
        joints_indices = range(njoints)

        cbs = [x for x in itertools.combinations(joints_indices, 3)]
        number_combinations = len(cbs)

        features = np.zeros((1, njoints * number_combinations))
        for i in range(njoints):
            for j in range(number_combinations):
                features[0, i * njoints + j] = \
                    M2IFeatures.compute_plane(
                        ske2[nx[cbs[j][0]]], ske2[ny[cbs[j][0]]], ske2[nz[cbs[j][0]]],
                        ske2[nx[cbs[j][1]]], ske2[ny[cbs[j][1]]], ske2[nz[cbs[j][1]]],
                        ske2[nx[cbs[j][2]]], ske2[ny[cbs[j][2]]], ske2[nz[cbs[j][2]]],
                        ske1[nx[i]], ske1[ny[i]], ske1[nz[i]])

        return features

    @staticmethod
    def normal_plane(ske1, ske2):
        n = ske1.shape[0]

        nx = range(0, n, 3)
        ny = range(1, n, 3)
        nz = range(2, n, 3)

        n_joints = len(nx)
        joints_indices = range(n_joints)
        cms = [x for x in itertools.combinations(joints_indices, 2)]
        number_combinations = len(cms)

        features = np.zeros((1, n_joints * number_combinations * (n_joints - 2)))
        for i in range(n_joints):
            for j in range(number_combinations):

                count = 0
                for k in range(n_joints):
                    if k in cms[j]:
                        continue

                    index = i*number_combinations*(n_joints - 2) + j*(n_joints - 2) + count
                    features[0, index] = \
                        M2IFeatures.compute_normal_plane(
                            ske2[nx[cms[j][0]]], ske2[ny[cms[j][0]]], ske2[nz[cms[j][0]]],
                            ske2[nx[cms[j][1]]], ske2[ny[cms[j][1]]], ske2[nz[cms[j][1]]],
                            ske2[nx[k]], ske2[ny[k]], ske2[nz[k]],
                            ske1[nx[i]], ske1[ny[i]], ske1[nz[i]])

                    count += 1

        return features

    @staticmethod
    def velocity_1(ske1, ske2, t):
        n = ske1.shape[0]

        nx = range(0, n, 3)
        ny = range(1, n, 3)
        nz = range(2, n, 3)

        n_joints = len(nx)
        joints_indices = range(n_joints)
        cms = [x for x in itertools.combinations(joints_indices, 2)]
        number_combinations = len(cms)

        features = np.zeros((1, number_combinations * (n_joints - 2) * 3))
        for j in range(number_combinations):

            count = 0
            for i in range(n_joints):
                if i in cms[j]:
                    continue

                v1, v2, v3 = M2IFeatures.find_velocity(
                    ske1[nx[i]], ske1[ny[i]], ske1[nz[i]],
                    ske2[nx[i]], ske2[ny[i]], ske2[nz[i]], t)

                ve1, ve2, ve3 = \
                    M2IFeatures.compute_velocity(
                        ske2[nx[cms[j][0]]], ske2[ny[cms[j][0]]], ske2[nz[cms[j][0]]],
                        ske2[nx[cms[j][1]]], ske2[ny[cms[j][1]]], ske2[nz[cms[j][1]]],
                        v1, v2, v3)

                index = j * (n_joints - 2) * 3 + count * 3
                features[0, index] = ve1
                features[0, index + 1] = ve2
                features[0, index + 2] = ve3
                count += 1

        return features

    @staticmethod
    def velocity_2(ske, ske1, ske2, t):
        n = ske1.shape[0]

        nx = range(0, n, 3)
        ny = range(1, n, 3)
        nz = range(2, n, 3)

        n_joints = len(nx)
        joints_indices = range(n_joints)
        cms = [x for x in itertools.combinations(joints_indices, 2)]
        number_combinations = len(cms)

        features = np.zeros((1, number_combinations * n_joints * 3))
        for j in range(number_combinations):

            count = 0
            for i in range(n_joints):
                v1, v2, v3 = M2IFeatures.find_velocity(
                    ske1[nx[i]], ske1[ny[i]], ske1[nz[i]],
                    ske2[nx[i]], ske2[ny[i]], ske2[nz[i]], t)

                ve1, ve2, ve3 = \
                    M2IFeatures.compute_velocity(
                        ske[nx[cms[j][0]]], ske[ny[cms[j][0]]], ske[nz[cms[j][0]]],
                        ske[nx[cms[j][1]]], ske[ny[cms[j][1]]], ske[nz[cms[j][1]]],
                        v1, v2, v3)

                index = j * (n_joints - 2) * 3 + count * 3
                features[0, index] = ve1
                features[0, index + 1] = ve2
                features[0, index + 2] = ve3
                count += 1

        return features

    @staticmethod
    def normal_velocity_1(ske1, ske2, t):
        n = ske1.shape[0]

        nx = range(0, n, 3)
        ny = range(1, n, 3)
        nz = range(2, n, 3)

        n_joints = len(nx)
        joints_indices = range(n_joints)
        cms = [x for x in itertools.combinations(joints_indices, 3)]
        number_combinations = len(cms)

        features = np.zeros((1, number_combinations * (n_joints - 3) * 3))
        for j in range(number_combinations):
            count = 0

            for i in range(n_joints):
                if i in cms[j]:
                    continue

                v1, v2, v3 = M2IFeatures.find_velocity(
                    ske1[nx[i]], ske1[ny[i]], ske1[nz[i]],
                    ske2[nx[i]], ske2[ny[i]], ske2[nz[i]], t)

                ve1, ve2, ve3 = \
                    M2IFeatures.compute_normal_velocity(
                        ske2[nx[cms[j][0]]], ske2[ny[cms[j][0]]], ske2[nz[cms[j][0]]],
                        ske2[nx[cms[j][1]]], ske2[ny[cms[j][1]]], ske2[nz[cms[j][1]]],
                        ske2[nx[cms[j][2]]], ske2[ny[cms[j][2]]], ske2[nz[cms[j][2]]],
                        v1, v2, v3)

                index = j * (n_joints - 3) * 3 + count * 3
                features[0, index] = ve1
                features[0, index + 1] = ve2
                features[0, index + 2] = ve3
                count += 1

        return features

    @staticmethod
    def normal_velocity_2(ske, ske1, ske2, t):
        n = ske1.shape[0]

        nx = range(0, n, 3)
        ny = range(1, n, 3)
        nz = range(2, n, 3)

        n_joints = len(nx)
        joints_indices = range(n_joints)
        cms = [x for x in itertools.combinations(joints_indices, 3)]
        number_combinations = len(cms)

        features = np.zeros((1, number_combinations * n_joints * 3))
        for j in range(number_combinations):
            count = 0

            for i in range(n_joints):
                v1, v2, v3 = M2IFeatures.find_velocity(
                    ske1[nx[i]], ske1[ny[i]], ske1[nz[i]],
                    ske2[nx[i]], ske2[ny[i]], ske2[nz[i]], t)

                ve1, ve2, ve3 = \
                    M2IFeatures.compute_normal_velocity(
                        ske[nx[cms[j][0]]], ske[ny[cms[j][0]]], ske[nz[cms[j][0]]],
                        ske[nx[cms[j][1]]], ske[ny[cms[j][1]]], ske[nz[cms[j][1]]],
                        ske[nx[cms[j][2]]], ske[ny[cms[j][2]]], ske[nz[cms[j][2]]],
                        v1, v2, v3)

                index = j * (n_joints - 3) * 3 + count * 3
                features[0, index] = ve1
                features[0, index + 1] = ve2
                features[0, index + 2] = ve3
                count += 1

        return features

    # -----------------
    # support function
    # -----------------
    @staticmethod
    def distance(x1, y1, z1, x2, y2, z2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

    @staticmethod
    def find_normal_vector(x1, y1, z1, x2, y2, z2, x3, y3, z3):
        vector1 = [x2 - x1, y2 - y1, z2 - z1]
        vector2 = [x3 - x1, y3 - y1, z3 - z1]

        cross_product = [vector1[1] * vector2[2] - vector1[2] * vector2[1],
                         vector1[2] * vector2[0] - vector1[0] * vector2[2],
                         vector1[0] * vector2[1] - vector1[1] * vector2[0]]

        return cross_product

    # plane constrains 3 points
    @staticmethod
    def find_plane(x1, y1, z1, x2, y2, z2, x3, y3, z3):
        cross_product = M2IFeatures.find_normal_vector(
            x1, y1, z1, x2, y2, z2, x3, y3, z3)

        a = cross_product[0]
        b = cross_product[1]
        c = cross_product[2]
        d = - (cross_product[0] * x1 + cross_product[1] * y1 + cross_product[2] * z1)

        return a, b, c, d

    # plane constrains 1 point and has 1 normal vector
    @staticmethod
    def find_plane_by_normal(x1, y1, z1, x2, y2, z2, x3, y3, z3):
        normal_vector = [x2 - x1, y2 - y1, z2 - z1]

        a = normal_vector[0]
        b = normal_vector[1]
        c = normal_vector[2]
        d = - (normal_vector[0] * x3 + normal_vector[1] * y3 + normal_vector[2] * z3)

        return a, b, c, d

    @staticmethod
    def plane_distance(x, y, z, a, b, c, d, euclidean_distance_abc):
        numerator = abs(x * a + y * b + c * z + d)
        return numerator / euclidean_distance_abc

    @staticmethod
    def compute_plane(x1, y1, z1, x2, y2, z2, x3, y3, z3, x, y, z):
        a, b, c, d = \
            M2IFeatures.find_plane(x1, y1, z1, x2, y2, z2, x3, y3, z3)
        euclidean_distance_abc = math.sqrt(a * a + b * b + c * c)
        distance = M2IFeatures.plane_distance(x, y, z, a, b, c, d, euclidean_distance_abc)

        return distance

    @staticmethod
    def compute_normal_plane(x1, y1, z1, x2, y2, z2, x3, y3, z3, x, y, z):
        a, b, c, d = \
            M2IFeatures.find_plane_by_normal(x1, y1, z1, x2, y2, z2, x3, y3, z3)
        euclidean_distance_abc = math.sqrt(a * a + b * b + c * c)
        distance = \
            M2IFeatures.plane_distance(x, y, z, a, b, c, d, euclidean_distance_abc)

        return distance

    @staticmethod
    def find_velocity(x1, y1, z1, x2, y2, z2, t):
        return (x2-x1)/t, (y2-y1)/t, (z2-z1)/t

    @staticmethod
    def compute_velocity(x1, y1, z1, x2, y2, z2, v1, v2, v3):
        d = M2IFeatures.distance(x1, y1, z1, x2, y2, z2)
        ve1 = v1 * (x2 - x1)
        ve2 = v2 * (y2 - y1)
        ve3 = v3 * (z2 - z1)

        return ve1/d, ve2/d, ve3/d

    @staticmethod
    def compute_normal_velocity(x1, y1, z1, x2, y2, z2, x3, y3, z3, v1, v2, v3):
        normal_vector = M2IFeatures.find_normal_vector(
            x1, y1, z1, x2, y2, z2, x3, y3, z3)
        ve1 = v1 * normal_vector[0]
        ve2 = v2 * normal_vector[1]
        ve3 = v3 * normal_vector[2]

        return ve1, ve2, ve3
