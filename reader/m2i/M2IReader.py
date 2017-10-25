import os
import sys
import cPickle
import numpy as np

from M2IFeaturesExtractor import *
from M2ISkeleton import *


class M2IReader(object):
    labels = ["Bow", "Box", "Chat", "Cross", "Handshake", "High Five", "Hug", "Wait", "Walk Together"]

    def __init__(self, dataset):
        self.dataset = dataset
        self.extractor = M2IFeaturesExtractor()

    def save_dataset_to(self, folder):
        list_thh1, list_thh2, list_shh, list_label, merged_ske1, merged_ske2 = self.extract()

        dataset = {'temporal_human_human_1': merged_ske1,
                   'temporal_human_human_2': merged_ske2,
                   'spatial_human_human': list_shh,
                   'labels': list_label}

        cPickle.dump(dataset, open('{1}/{0}.pik'.format("m2i_dataset", folder), 'wb'))

    def extract(self):
        list_ske1, list_ske2, list_labels = self.read()

        list_thh1, list_thh2, list_shh, merged_ske1, merged_ske2 = self.extractor.get_features(list_ske1, list_ske2)

        labels = []
        for i in range(len(list_labels)):
            for j in range(len(list_labels[i])):
                lbl = np.asarray(list_labels[i][j])
                lbl = lbl.reshape((len(list_labels[i][j]), 1))
                labels.append(lbl)

        return list_thh1, list_thh2, list_shh, labels, merged_ske1, merged_ske2

    def read(self):
        activities = os.listdir(self.dataset)

        list_ske1 = []
        list_ske2 = []
        list_label = []

        for activity in activities:
            path = self.dataset + activity + '/'
            if os.path.isdir(path):
                ske1, ske2, lbl = M2IReader.read_activity(path, M2IReader.get_label(activity))

                list_ske1.append(ske1)
                list_ske2.append(ske2)
                list_label.append(lbl)

        return list_ske1, list_ske2, list_label

    @staticmethod
    def read_activity(activity, label):
        views = os.listdir(activity)

        list_ske1 = []
        list_ske2 = []
        list_label = []

        for view in views:
            path = activity + view + '/'
            if os.path.isdir(path):
                ske1, ske2, lbl = M2IReader.read_view(path, label)
                list_ske1.append(ske1)
                list_ske2.append(ske2)
                list_label.append(lbl)

        return list_ske1, list_ske2, list_label

    @staticmethod
    def read_view(view, label):
        scenes = os.listdir(view)

        list_ske1 = []
        list_ske2 = []
        list_label = []

        for scene in scenes:
            path = view + scene

            ske1, ske2 = M2IReader.read_scene(path)

            list_ske1.append(ske1)
            list_ske2.append(ske2)
            list_label.append(label)

        return list_ske1, list_ske2, list_label

    @staticmethod
    def read_scene(scene):
        ske1, ske2 = M2ISkeleton.read(scene)

        return ske1, ske2

    @staticmethod
    def get_label(label):
        return M2IReader.labels.index(label)
