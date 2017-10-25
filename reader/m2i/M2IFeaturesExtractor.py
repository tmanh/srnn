from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from M2IFeatures import *
from M2ISkeleton import *


class M2IFeaturesExtractor(object):
    temporal_features_length = 400
    spatial_features_length = 400
    max_time_steps = 171

    def __init__(self):
        self.m2i_features_utils = M2IFeatures()

    def get_features(self, skeleton1, skeleton2):
        # joints distance = between 2 people (225) + 1 person (105 * 2)
        # Joint motion = between 2 people (225) + 1 person (105 * 2)
        # joint_displacement = 1 person (45*2)
        # plane = between 2 people (6825) + 1 person (105 * 2)

        list_thh1, list_thh2, list_shh, merged_ske1, merged_ske2 = self.extract_features(skeleton1, skeleton2)

        # joint_distance = self.sbu_features_utils.joint_distance(ske1=ske1, ske2=ske2)
        # joint_displacement = self.sbu_features_utils.joint_displacement(ske1=ske1, ske2=ske2)
        # plane = self.sbu_features_utils.plane(ske1=ske1, ske2=ske2)
        # normal_planes = self.sbu_features_utils.normal_plane(ske1=ske1, ske2=ske2)
        # velocity1 = self.sbu_features_utils.velocity_1(ske1=ske1, ske2=ske2, t=1)
        # normal_velocity1 = self.sbu_features_utils.normal_velocity_1(ske1=ske1, ske2=ske2, t=1)

        # self.display_skeleton(ske1)

        return list_thh1, list_thh2, list_shh, merged_ske1, merged_ske2

    def extract_features(self, skeleton1, skeleton2):
        list_all_thh1 = []
        list_all_thh2 = []
        list_all_shh = []
        merged_ske1 = []
        merged_ske2 = []

        for case_ske_1, case_ske_2 in zip(skeleton1, skeleton2):
            view_thh1 = []
            view_thh2 = []
            view_shh = []

            for view_ske1, view_ske2 in zip(case_ske_1, case_ske_2):
                list_thh1 = []
                list_thh2 = []
                list_shh = []

                for activity_ske1, activity_ske2 in zip(view_ske1, view_ske2):
                    thh1 = self.temporal_human_human_features(activity_ske1)
                    thh2 = self.temporal_human_human_features(activity_ske2)
                    shh = self.spatial_human_human_features(activity_ske1, activity_ske2)

                    list_thh1.append(thh1)
                    list_thh2.append(thh2)
                    list_shh.append(shh)

                view_thh1.append(list_thh1)
                view_thh2.append(list_thh2)
                view_shh.append(list_shh)

            for i in range(len(view_thh1)):
                thh1_features, thh2_features, shh_features, ske_1, ske_2 = \
                    self.concat_features(view_thh1[i], view_thh2[i], view_shh[i], case_ske_1[i], case_ske_1[i])

                list_all_thh1.append(thh1_features)
                list_all_thh2.append(thh2_features)
                list_all_shh.append(shh_features)
                merged_ske1.append(ske_1)
                merged_ske2.append(ske_2)

        return list_all_thh1, list_all_thh2, list_all_shh, merged_ske1, merged_ske2

    def concat_features(self, list_thh1, list_thh2, list_shh, ske1, ske2):
        thh1 = np.zeros((len(list_thh1),
                         self.max_time_steps,
                         self.temporal_features_length))
        thh2 = np.zeros((len(list_thh1),
                         self.max_time_steps,
                         self.temporal_features_length))
        shh = np.zeros((len(list_thh1),
                        self.max_time_steps,
                        self.spatial_features_length))
        merged_ske1 = np.zeros((len(list_thh1),
                                self.max_time_steps,
                                60))
        merged_ske2 = np.zeros((len(list_thh1),
                                self.max_time_steps,
                                60))

        for i in range(len(list_thh1)):
            thh1[i, self.max_time_steps - list_thh1[i].shape[0]:, :] = list_thh1[i]
            thh2[i, self.max_time_steps - list_thh1[i].shape[0]:, :] = list_thh2[i]
            shh[i, self.max_time_steps - list_thh1[i].shape[0]:, :] = list_shh[i]
            merged_ske1[i, self.max_time_steps - list_thh1[i].shape[0]:, :] = ske1[i]
            merged_ske2[i, self.max_time_steps - list_thh1[i].shape[0]:, :] = ske2[i]

        return thh1, thh2, shh, merged_ske1, merged_ske2

    def temporal_human_human_features(self, ske):
        features = np.zeros((ske.shape[0], self.temporal_features_length))
        for i in range(1, len(ske)):
            features[i, :] = M2IFeatures.joint_distance(ske[i-1, :], ske[i, :])
        return features

    def spatial_human_human_features(self, ske1, ske2):
        features = np.zeros((ske1.shape[0], self.spatial_features_length))
        for i in range(len(ske1)):
            features[i, :] = M2IFeatures.joint_distance(ske1[i, :], ske2[i, :])
        return features

    def find_max_temporal_steps(self, list_feature):
        max_steps = 0
        for features in list_feature:
            max_steps = max(features.shape[0], max_steps)
        return max_steps
