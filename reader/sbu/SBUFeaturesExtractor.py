from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from SBUFeatures import *
from SBUSkeleton import *


class SBUFeaturesExtractor(object):
    temporal_features_length = 225
    spatial_features_length = 225
    max_time_steps = 46  # 46 for clean, 52 for noisy

    def __init__(self):
        self.sbu_features_utils = SBUFeatures()

    def get_features(self, skeleton1, skeleton2):
        # joints distance = between 2 people (225) + 1 person (105 * 2)
        # Joint motion = between 2 people (225) + 1 person (105 * 2)
        # joint_displacement = 1 person (45*2)
        # plane = between 2 people (6825) + 1 person (105 * 2)

        list_thh1, list_thh2, list_shh = self.extract_features(skeleton1, skeleton2)

        # joint_distance = self.sbu_features_utils.joint_distance(ske1=ske1, ske2=ske2)
        # joint_displacement = self.sbu_features_utils.joint_displacement(ske1=ske1, ske2=ske2)
        # plane = self.sbu_features_utils.plane(ske1=ske1, ske2=ske2)
        # normal_planes = self.sbu_features_utils.normal_plane(ske1=ske1, ske2=ske2)
        # velocity1 = self.sbu_features_utils.velocity_1(ske1=ske1, ske2=ske2, t=1)
        # normal_velocity1 = self.sbu_features_utils.normal_velocity_1(ske1=ske1, ske2=ske2, t=1)

        # self.display_skeleton(ske1)

        return list_thh1, list_thh2, list_shh

    def extract_features(self, skeleton1, skeleton2):
        list_all_thh1 = []
        list_all_thh2 = []
        list_all_shh = []

        for case_ske_1, case_ske_2 in zip(skeleton1, skeleton2):
            list_thh1 = []
            list_thh2 = []
            list_shh = []

            for activity_ske1, activity_ske2 in zip(case_ske_1, case_ske_2):
                thh1 = self.temporal_human_human_features(activity_ske1)
                thh2 = self.temporal_human_human_features(activity_ske2)
                shh = self.spatial_human_human_features(activity_ske1, activity_ske2)

                list_thh1.append(thh1)
                list_thh2.append(thh2)
                list_shh.append(shh)

            thh1_features, thh2_features, shh_features = \
                self.concat_features(list_thh1, list_thh2, list_shh)

            list_all_thh1.append(thh1_features)
            list_all_thh2.append(thh2_features)
            list_all_shh.append(shh_features)

        return list_all_thh1, list_all_thh2, list_all_shh

    def concat_features(self, list_thh1, list_thh2, list_shh):
        thh1 = np.zeros((len(list_thh1),
                         self.max_time_steps,
                         self.temporal_features_length))
        thh2 = np.zeros((len(list_thh1),
                         self.max_time_steps,
                         self.temporal_features_length))
        shh = np.zeros((len(list_thh1),
                        self.max_time_steps,
                        self.spatial_features_length))

        for i in range(len(list_thh1)):
            thh1[i, self.max_time_steps - list_thh1[i].shape[0]:, :] = list_thh1[i]
            thh2[i, self.max_time_steps - list_thh1[i].shape[0]:, :] = list_thh2[i]
            shh[i, self.max_time_steps - list_thh1[i].shape[0]:, :] = list_shh[i]

        return thh1, thh2, shh

    def temporal_human_human_features(self, ske):
        features = np.zeros((ske.shape[0], self.temporal_features_length))
        for i in range(1, len(ske)):
            features[i, :] = SBUFeatures.joint_distance(ske[i-1, :], ske[i, :])
        return features

    def spatial_human_human_features(self, ske1, ske2):
        features = np.zeros((ske1.shape[0], self.spatial_features_length))
        for i in range(len(ske1)):
            features[i, :] = SBUFeatures.joint_distance(ske1[i, :], ske2[i, :])
        return features

    def find_max_temporal_steps(self, list_feature):
        max_steps = 0
        for features in list_feature:
            max_steps = max(features.shape[0], max_steps)
        return max_steps

    # 0: HEAD
    # 1: NECK
    # 2: TRUNK
    # 3: LEFT SHOULDER
    # 4: LEFT ELBOW
    # 5: LEFT HAND
    # 6: RIGHT SHOULDER
    # 7: RIGHT ELBOW
    # 8: RIGHT HAND
    # 9: LEFT HIP
    # 10: LEFT KNEE
    # 11: LEFT FOOT
    # 12: RIGHT HIP
    # 13: RIGHT KNEE
    # 14: RIGHT FOOT
    @staticmethod
    def display_skeleton(ske):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        nx = range(0, 45, 3)
        ny = range(1, 45, 3)
        nz = range(2, 45, 3)

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

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()
