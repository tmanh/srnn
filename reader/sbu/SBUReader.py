import os
import sys
import cPickle
import numpy as np

from SBUFeaturesExtractor import *
from SBUSkeleton import *


class SBUReader(object):
    labels = ["01", "02", "03", "04", "05", "06", "07", "08"]

    def __init__(self, dataset):
        self.dataset = dataset
        self.extractor = SBUFeaturesExtractor()

    def save_dataset_to(self, folder):
        list_thh1, list_thh2, list_shh, list_label = self.extract()

        dataset = {'temporal_human_human_1': list_thh1,
                   'temporal_human_human_2': list_thh2,
                   'spatial_human_human': list_shh,
                   'labels': list_label}

        cPickle.dump(dataset, open('{1}/{0}.pik'.format("sbu_dataset", folder), 'wb'))

    def extract(self):
        list_ske1, list_ske2, list_labels = self.read()

        list_thh1, list_thh2, list_shh = self.extractor.get_features(list_ske1, list_ske2)

        labels = []
        for i in range(len(list_labels)):
            lbl = np.asarray(list_labels[i])
            lbl = lbl.reshape((len(list_labels[i]), 1))
            labels.append(lbl)

        return list_thh1, list_thh2, list_shh, labels

    def read(self):
        actors = os.listdir(self.dataset)

        list_ske1 = []
        list_ske2 = []
        list_label = []

        for actor in actors:
            path = self.dataset + actor + '/'
            if os.path.isdir(path):
                ske1, ske2, lbl = SBUReader.read_actor(path)

                list_ske1.append(ske1)
                list_ske2.append(ske2)
                list_label.append(lbl)

        return list_ske1, list_ske2, list_label

    @staticmethod
    def read_actor(actor):
        activities = os.listdir(actor)

        list_ske1 = []
        list_ske2 = []
        list_label = []

        for activity in activities:
            path = actor + activity + '/'
            if os.path.isdir(path):
                ske1, ske2, lbl = SBUReader.read_activity(path, SBUReader.get_label(activity))

                list_ske1.extend(ske1)
                list_ske2.extend(ske2)
                list_label.extend(lbl)

        return list_ske1, list_ske2, list_label

    @staticmethod
    def read_activity(activity, label):
        scenes = os.listdir(activity)

        list_ske1 = []
        list_ske2 = []
        list_label = []

        for scene in scenes:
            path = activity + scene + '/'
            if os.path.isdir(path):
                ske1, ske2 = SBUReader.read_scene(path)
                list_ske1.append(ske1)
                list_ske2.append(ske2)
                list_label.append(label)

        return list_ske1, list_ske2, list_label

    @staticmethod
    def read_scene(scene):
        skeleton_path = scene + 'skeleton_pos.txt'

        ske1, ske2 = SBUSkeleton.read(skeleton_path)

        return ske1, ske2

    @staticmethod
    def get_label(label):
        return SBUReader.labels.index(label)
