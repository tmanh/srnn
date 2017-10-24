# This code is modified from https://github.com/asheshjain399/RNNexp
# Many thanks to Ashesh Jain for the reading CAD-120 features code
#
# [1] Learning Human Activities and Object Affordances from RGB-D Videos, Hema S Koppula, Rudhir Gupta, Ashutosh Saxena.
#     International Journal of Robotics Research (IJRR), in press, Jan 2013.
#
# Anh Minh Truong
# Email: tmanh93@outlook.com

import numpy as np
import re
from os import listdir
import random
import cPickle
import sys
import os
import pdb

node_t_o_o_dim = 0
node_s_o_o_dim = 0
node_t_h_h_dim = 0
node_s_h_o_dim = 0

edge_t_o_o_dim = 0
edge_t_h_h_dim = 0


def sample_subsequences(length, num_samples=1, min_len=1, max_len=10):
    max_len = min(max_len, length)
    min_len = min(min_len, max_len)
    sequence = []
    for i in range(num_samples):
        ll = random.randint(min_len, max_len)
        start_idx = random.randint(0, length - ll)
        end_idx = start_idx + ll
        if not (start_idx, end_idx) in sequence:
            sequence.append((start_idx, end_idx))
    return sequence


def parse_feature_vector(features_list):
    f_list = [int(x.split(':')[1]) for x in features_list]
    return f_list


# read the features extracted from http://pr.cs.cornell.edu/humanactivities/data.php
def get_node_info(folder, filename):
    f = open(folder + '/' + filename, 'r')
    header = f.readline()
    f.close()

    node_info = [int(x) for x in header.strip().split(' ')]
    num_objects = node_info[0]
    num_o_o_edges = node_info[1]
    num_s_o_edges = node_info[2]

    return num_objects, num_o_o_edges, num_s_o_edges


def get_edge_info(folder, filename):
    f = open(folder + '/' + filename, 'r')
    header = f.readline()
    f.close()

    edge_info = [int(x) for x in header.strip().split(' ')]
    num_o_o_edge = edge_info[0]

    return num_o_o_edge


def parse_temporal_object_object_features(features_list, pos, length):
    global node_t_o_o_dim

    o_aff = []
    t_o_o_ids = []
    t_o_o_fea = []

    for l in features_list[pos:pos + length]:
        split_str = l.strip().split(' ')
        o_aff.append(int(split_str[0]))
        t_o_o_ids.append(int(split_str[1]))
        t_o_o_fea.append(parse_feature_vector(split_str[2:]))
        node_t_o_o_dim = int(len(t_o_o_fea[-1]))

    pos = pos + length
    return o_aff, t_o_o_ids, t_o_o_fea, pos


def parse_spatial_object_object_features(features_list, pos, length):
    global node_s_o_o_dim

    s_o_o_ids = []
    s_o_o_fea = []

    for l in features_list[pos:pos + length]:
        split_str = l.strip().split(' ')
        s_o_o_ids.append([int(split_str[2]), int(split_str[3])])
        s_o_o_fea.append(parse_feature_vector(split_str[4:]))
        node_s_o_o_dim = int(2 * len(s_o_o_fea[-1]))

    pos = pos + length
    return s_o_o_ids, s_o_o_fea, pos


def parse_temporal_human_human_features(features_list, pos):
    global node_t_h_h_dim

    skeleton_info = features_list[pos].strip().split(' ')
    t_h_h_id = int(skeleton_info[0])
    t_h_h_fea = parse_feature_vector(skeleton_info[2:])

    node_t_h_h_dim = len(t_h_h_fea)

    pos = pos + 1
    return t_h_h_id, t_h_h_fea, pos


def parse_spatial_human_object_features(features_list, pos, length):
    global node_s_h_o_dim

    s_h_o_ids = []
    s_h_o_fea = []

    for l in features_list[pos: pos + length]:
        split_str = l.strip().split(' ')
        s_h_o_ids.append(int(split_str[2]))
        s_h_o_fea.append(parse_feature_vector(split_str[3:]))
        node_s_h_o_dim = len(s_h_o_fea[-1])

    pos = pos + length
    return s_h_o_ids, s_h_o_fea, pos


# read node features as my S-RNN format
def read_node_feature(folder, filename):
    num_objects, num_o_o_edges, num_s_o_edges = get_node_info(folder, filename)

    f = open(folder + '/' + filename, 'r')
    list_fea = f.readlines()[1:]
    f.close()

    pos = 0

    o_aff, t_o_o_ids, t_o_o_fea, pos = parse_temporal_object_object_features(list_fea, pos, num_objects)
    t_h_h_id, t_h_h_fea, pos = parse_temporal_human_human_features(list_fea, pos)
    s_o_o_ids, s_o_o_fea, pos = parse_spatial_object_object_features(list_fea, pos, num_o_o_edges)
    s_h_o_ids, s_h_o_fea, pos = parse_spatial_human_object_features(list_fea, pos, num_s_o_edges)

    return {
        'sub_activity': t_h_h_id,
        'object_affordances': o_aff,
        't_h_h_fea': t_h_h_fea,
        't_o_o_ids': t_o_o_ids,
        't_o_o_fea': t_o_o_fea,
        's_o_o_ids': s_o_o_ids,
        's_o_o_fea': s_o_o_fea,
        's_h_o_ids': s_h_o_ids,
        's_h_o_fea': s_h_o_fea
    }


# convert edge features as my S-RNN format
def read_edge_feature(folder, filename):
    num_o_o_edge = get_edge_info(folder, filename)

    f = open(folder + '/' + filename, 'r')
    features_list = f.readlines()[1:]
    f.close()

    t_o_o_ids = []
    t_o_o_fea = []
    for l in features_list[0:num_o_o_edge]:
        split_str = l.strip().split(' ')
        t_o_o_ids.append(int(split_str[2]))
        t_o_o_fea.append(parse_feature_vector(split_str[3:]))

    skeleton_stats = features_list[num_o_o_edge].strip().split(' ')
    t_h_h_fea = parse_feature_vector(skeleton_stats[3:])

    return {
        't_o_o_ids': t_o_o_ids,
        't_o_o_fea': t_o_o_fea,
        't_h_h_fea': t_h_h_fea
    }


def init_activity_dict(folder, fin):
    features_node = {}
    features_node_node = {}
    features_temporal_edge = {}
    target = {}

    num_o, num_o_o_e, num_s_o_e = get_node_info(folder, fin)

    target['h'] = []
    features_node['h'] = []
    features_temporal_edge['h'] = []
    features_node_node['h'] = {}

    for obj1 in range(1, num_o + 1):
        features_node[str(obj1)] = []
        features_temporal_edge[str(obj1)] = []
        features_node_node['h'][str(obj1)] = []
        features_node_node[str(obj1)] = {}
        target[str(obj1)] = []

        for obj2 in range(1, num_o + 1):
            if obj1 == obj2:
                continue
            features_node_node[str(obj1)][str(obj2)] = []

    return features_node, features_node_node, features_temporal_edge, target


def is_node_file(fin):
    return len(fin.split('_')) == 2


def is_edge_file(fin):
    return len(fin.split('_')) == 3


def read_node_file(folder, f, target, features_node, features_node_node):
    key_value = read_node_feature(folder, f)

    features_node['h'].append(key_value['t_h_h_fea'])
    target['h'].append(key_value['sub_activity'])

    for i in range(len(key_value['t_o_o_ids'])):
        t_o_o_ids = key_value['t_o_o_ids'][i]
        t_o_o_fea = key_value['t_o_o_fea'][i]
        object_affordances = key_value['object_affordances'][i]
        features_node[str(t_o_o_ids)].append(t_o_o_fea)
        target[str(t_o_o_ids)].append(object_affordances)

    for i in range(len(key_value['s_h_o_ids'])):
        s_h_o_ids = key_value['s_h_o_ids'][i]
        s_h_o_fea = key_value['s_h_o_fea'][i]
        features_node_node['h'][str(s_h_o_ids)].append(s_h_o_fea)

    for i in range(len(key_value['s_o_o_ids'])):
        s_o_o_ids = key_value['s_o_o_ids'][i]
        s_o_o_fea = key_value['s_o_o_fea'][i]
        features_node_node[str(s_o_o_ids[0])][str(s_o_o_ids[1])].append(s_o_o_fea)


def read_edge_file(folder, f, features_temporal_edge):
    global edge_t_h_h_dim, edge_t_o_o_dim

    key_value = read_edge_feature(folder, f)

    features_temporal_edge['h'].append(key_value['t_h_h_fea'])
    edge_t_h_h_dim = int(len(key_value['t_h_h_fea']))

    for i in range(len(key_value['t_o_o_ids'])):
        t_o_o_ids = key_value['t_o_o_ids'][i]
        t_o_o_fea = key_value['t_o_o_fea'][i]
        features_temporal_edge[str(t_o_o_ids)].append(t_o_o_fea)
        edge_t_o_o_dim = int(len(t_o_o_fea))


# read the features from node files and edge files (extracted from [1])
def read_activity(folder, files):
    features_node, features_node_node, features_temporal_edge, target = init_activity_dict(folder, files[0])

    for f in files:
        if is_node_file(f):
            read_node_file(folder, f, target, features_node, features_node_node)
        elif is_edge_file(f):
            read_edge_file(folder, f, features_temporal_edge)

    # add padding time step
    for k in features_temporal_edge.keys():
        features_temporal_edge[k].insert(0, [0] * len(features_temporal_edge[k][0]))

    # convert to array
    for k in target.keys():
        target[k] = np.array(target[k])

    # convert to array
    for k in features_node:
        features_node[k] = np.array(features_node[k])

    # remove redundant temporal edges and align time steps
    for k in features_temporal_edge.keys():
        features_temporal_edge[k] = np.array(features_temporal_edge[k])

        if not (features_node['h'].shape[0] == features_temporal_edge[k].shape[0]):
            features_temporal_edge[k] = features_temporal_edge[k][:-1]

    # convert to array
    for k in features_node_node.keys():
        for k2 in features_node_node[k].keys():
            features_node_node[k][k2] = np.array(features_node_node[k][k2])

    # check shapes
    for k in features_temporal_edge.keys():
        assert (features_node['h'].shape[0] == features_temporal_edge[k].shape[0])

    return target, features_node, features_temporal_edge, features_node_node


def all_activities_info(all_the_files):
    all_activities = []
    activities_time_steps = {}

    for f in all_the_files:
        split_str = f.split('_')[0]
        if split_str not in all_activities:
            all_activities.append(split_str)
            activities_time_steps[split_str] = 1
        else:
            activities_time_steps[split_str] += 1

    time_steps = int((max(activities_time_steps.values()) + 1) / 2)
    print 'max time ', time_steps

    return all_activities, time_steps


def read_high_level_activity_groundtruth(fin):
    f = open(fin, 'r')
    ground_truth = f.readlines()
    f.close()

    high_level_activity_groundtruth = {}

    for i in range(len(ground_truth)):
        split_str = [x for x in ground_truth[i].strip().split(' ')]
        high_level_activity_groundtruth[split_str[0]] = int(split_str[1])

    return high_level_activity_groundtruth


# input is node_object features
def find_maximum_objects(features_list):
    max_objects = 0

    for f in features_list:
        if max_objects < len(f):
            max_objects = len(f)

    return max_objects


def generate_temporal_human_human_features(features_list, dim, time_steps):
    n_samples = len(features_list[0])
    thh_dim = 0
    for d in dim:
        thh_dim += d

    thh_features = np.zeros((n_samples, time_steps, thh_dim), dtype=np.float32)

    for i in range(n_samples):
        d_start = 0

        for f, d in zip(features_list, dim):
            t = f[i].shape[0]
            thh_features[i:i + 1, time_steps - t:, d_start:d_start + d] = np.reshape(f[i], (1, t, d))
            d_start += d

    return thh_features


def generate_temporal_object_object_features(features_list, dim, time_steps, max_objects):
    n_samples = len(features_list[0])
    too_dim = 0
    for d in dim:
        too_dim += d

    too_features = np.zeros((n_samples, max_objects, time_steps, too_dim), dtype=np.float32)

    for i in range(n_samples):
        d_start = 0
        for f, d in zip(features_list, dim):
            for o in range(len(f[i])):
                t = f[i][o].shape[0]

                too_features[i:i + 1, o, time_steps - t:, d_start:d_start + d] = np.reshape(f[i][o], (1, 1, t, d))
            d_start += d

    return too_features


def generate_spatial_features(features_list, dim, time_steps, max_objects):
    n_samples = len(features_list)

    s_features = np.zeros((n_samples, max_objects, time_steps, dim), dtype=np.float32)

    for i in range(n_samples):
        for o in range(len(features_list[i])):
            t = features_list[i][o].shape[0]

            if features_list[i][o].shape[1] == 0:
                continue

            s_features[i:i + 1, o, time_steps - t:, 0:dim] = np.reshape(features_list[i][o], (1, 1, t, dim))

    return s_features


def save_data(save_path, time_steps, folder, activities, high_level_activity_groundtruth):
    [y_activity, y, node, edge, y_object, node_object, edge_object, edge_intra_object, edge_intra_object_human] = \
        append_features(folder, activities, high_level_activity_groundtruth)

    max_objects = find_maximum_objects(node_object)

    [target_activity, target, target_object, target_anticipation, target_object_anticipation] = \
        generate_labels(y_activity, y, y_object, time_steps, max_objects)

    thh_features = generate_temporal_human_human_features([node, edge], [node_t_h_h_dim, edge_t_h_h_dim], time_steps)
    too_features = generate_temporal_object_object_features(
        [node_object, edge_object], [node_t_o_o_dim, edge_t_o_o_dim], time_steps, max_objects)
    soo_features = generate_spatial_features(edge_intra_object, node_s_o_o_dim, time_steps, max_objects)
    soh_features = generate_spatial_features(edge_intra_object_human, node_s_h_o_dim, time_steps, max_objects)

    data = {'labels_activity': target_activity, 'labels_human': target, 'labels_objects': target_object,
            'labels_human_anticipation': target_anticipation,
            'labels_objects_anticipation': target_object_anticipation,
            'thh_features': thh_features, 'too_features': too_features,
            'soo_features': soo_features, 'soh_features': soh_features}
    cPickle.dump(data, open(save_path, 'wb'))


def gather_all_files(activity, all_files):
    idx = 1
    filenames = []

    while True:
        f = '{0}_{1}.txt'.format(activity, idx)
        if f not in all_files:
            break
        filenames.append(f)

        f = '{0}_{1}_{2}.txt'.format(activity, idx, idx + 1)
        if f not in all_files:
            break
        filenames.append(f)

        idx = idx + 1

    return filenames


def get_labels_and_features_activity(target, node, temporal_edge,
                                     all_target_human, all_node_human, all_temporal_edge_human):
    all_target_human.append(target['h'])
    all_node_human.append(node['h'])
    all_temporal_edge_human.append(temporal_edge['h'])


def get_labels_and_features_affordances(target, node, temporal_edge, intra_edge,
                                        all_target_object, all_node_object, all_temporal_edge_object,
                                        all_intra_object_object, all_intra_object_human):
    target_object = []
    node_object = []
    temporal_edge_object = []
    intra_object_object = []
    intra_object_human = []

    object_ids = target.keys()
    del object_ids[object_ids.index('h')]

    for oid in object_ids:
        target_object.append(target[oid])
        node_object.append(node[oid])
        temporal_edge_object.append(temporal_edge[oid])
        intra_object_human.append(intra_edge['h'][oid])

        intra_o_o = np.zeros((node[oid].shape[0], node_s_o_o_dim))
        for _oid in object_ids:
            if _oid == oid:
                continue
            intra_o_o[:, :intra_edge[oid][_oid].shape[1]] += intra_edge[oid][_oid]
            intra_o_o[:, intra_edge[oid][_oid].shape[1]:] += intra_edge[_oid][oid]
        intra_object_object.append(intra_o_o)

    all_target_object.append(target_object)
    all_node_object.append(node_object)
    all_temporal_edge_object.append(temporal_edge_object)
    all_intra_object_object.append(intra_object_object)
    all_intra_object_human.append(intra_object_human)


def append_features(folder, all_activities, high_level_activity_groundtruth):
    all_files = listdir(folder)

    # labels and features of human activity
    target_activity = []  # high-level activity
    target_human = []
    node_human = []
    temporal_edge_human = []

    # labels and features of object affordances
    target_object = []
    node_object = []
    temporal_edge_object = []
    intra_object_object = []
    intra_object_human = []

    for activity in all_activities:
        target_activity.append(high_level_activity_groundtruth[activity])

        filenames = gather_all_files(activity, all_files)

        target, node, temporal_edge, intra_edge = read_activity(folder, filenames)

        get_labels_and_features_activity(target, node, temporal_edge,
                                         target_human, node_human, temporal_edge_human)

        get_labels_and_features_affordances(target, node, temporal_edge, intra_edge,
                                            target_object, node_object, temporal_edge_object,
                                            intra_object_object, intra_object_human)

    return [target_activity, target_human, node_human, temporal_edge_human,
            target_object, node_object, temporal_edge_object, intra_object_object, intra_object_human]


def generate_anticipate_labels(y, y_object):
    y_anticipation = []
    y_object_anticipation = []

    for l in y:
        y_anticipation.append(append2array(l[1:], 11))

    for yo in y_object:
        y_object_anticipation_ = []

        for yo_ in yo:
            y_object_anticipation_.append(append2array(yo_[1:], 13))

        y_object_anticipation.append(y_object_anticipation_)

    return y_anticipation, y_object_anticipation


def generate_labels(y_activity, y, y_object, time_steps, max_objects):
    n_samples = len(y)

    y_anticipation, y_object_anticipation = generate_anticipate_labels(y, y_object)

    target_activity = np.zeros(n_samples, dtype=np.int32)
    target = np.zeros((n_samples, time_steps), dtype=np.int32)
    target_object = np.zeros((n_samples, max_objects, time_steps), dtype=np.int32)
    target_anticipation = np.zeros((n_samples, time_steps), dtype=np.int32)
    target_object_anticipation = np.zeros((n_samples, max_objects, time_steps), dtype=np.int32)

    for i in range(n_samples):
        target_activity[i] = y_activity[i]

        t = y[i].shape[0]
        target[i:i+1, time_steps-t:] = y[i]
        target_anticipation[i:i + 1, time_steps - t:] = np.reshape(y_anticipation[i], (1, t))

        for o in range(len(y_object[i])):
            target_object[i:i + 1, o, time_steps - t:] = np.reshape(y_object[i][o], (1, 1, t))
            target_object_anticipation[i:i + 1, o, time_steps - t:] = np.reshape(y_object_anticipation[i][o], (1, 1, t))

    return [target_activity, target, target_object, target_anticipation, target_object_anticipation]


def append2array(a, add, choose_list=None):
    ll = list(a)
    ll.append(add)
    temp_array = np.array(ll)
    if choose_list:
        temp_array = temp_array[choose_list]
    return temp_array


def save_data2files(folder, train_activities, test_activities):
    high_level_activities = '/home/anhtruong/Workspace/srnn/data/features_cad120/high-level-activity_groundtruth.txt'
    dataset = '/home/anhtruong/Workspace/srnn/data/CAD-120/fold_{0}'.format(fold)

    high_level_activity_groundtruth = read_high_level_activity_groundtruth(high_level_activities)

    if not os.path.exists(dataset):
        os.mkdir(dataset)

    all_the_files = listdir(folder)
    _, time_steps = all_activities_info(all_the_files)

    save_data('{0}/train_data.pik'.format(dataset),
              time_steps, folder, train_activities, high_level_activity_groundtruth)
    save_data('{0}/test_data.pik'.format(dataset),
              time_steps, folder, test_activities, high_level_activity_groundtruth)


if __name__ == '__main__':
    train_activities = []
    test_activities = []

    folds = ['1', '2', '3', '4']
    for fold in folds:
        s = '/home/anhtruong/Workspace/srnn/data/features_cad120/features_binary_svm_format'
        test_file = '/home/anhtruong/Workspace/srnn/data/features_cad120/activityids/activityids_fold{0}.txt'.format(
            fold)

        test_activities = []
        lines = open(test_file).readlines()
        for line in lines:
            line = line.strip()
            if len(line) > 0:
                test_activities.append(line)
        print "test ", test_file

        train_activities = []
        for j in folds:
            if j == fold:
                continue
            train_file = '/home/anhtruong/Workspace/srnn/data/features_cad120/activityids/activityids_fold{0}.txt'.format(
                j)
            print "train ", train_file
            lines = open(train_file).readlines()
            for line in lines:
                line = line.strip()
                if len(line) > 0:
                    train_activities.append(line)

        print len(train_activities)
        print len(test_activities)

        N = len(train_activities) + len(test_activities)
        print N

        save_data2files(s, train_activities, test_activities)
