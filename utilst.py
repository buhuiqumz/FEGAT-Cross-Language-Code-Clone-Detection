import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve,f1_score,accuracy_score,precision_score,recall_score
# from graphnnSiamese1 import graphnn1
import json
import pandas as pd
import os


# def get_f_name(DATA, SF, CM, OP, VS):
#     F_NAME = []
#     for sf in SF:#SOFTWARE
#         for cm in CM:#COMPILER
#             for op in OP:#OPTIMIZATION
#                 for vs in VS:#VERSION
#                     F_NAME.append(DATA+sf+cm+op+vs+".json")
#     return F_NAME

def build_edges_matrix(edges,nodes_number_limit):

    edges_matric = [[], []]
    for edge in edges:

        source_vertex_id = edge[0]
        target_vertex_id = edge[1]
        if (source_vertex_id<nodes_number_limit) and (target_vertex_id<nodes_number_limit ):
            edges_matric[0].append(source_vertex_id)
            edges_matric[1].append(target_vertex_id)

    return edges_matric

def ensure_permutation_file(classes):
    if os.path.isfile('data/class_perm.npy'):
        perm = np.load('data/class_perm.npy')
    else:
        perm = np.random.permutation(len(classes))
        np.save('data/class_perm.npy', perm)
    if len(perm) < len(classes):
        perm = np.random.permutation(len(classes))
        np.save('data/class_perm.npy', perm)
    return perm


def get_f_name():
    F_NAME = []

    F_NAME.append('java-cpp-9-9.json')
    return F_NAME


def get_f_dict(F_NAME):
    name_num = 0
    name_dict = {}

    for f_name in F_NAME:
        with open(f_name, 'r', encoding='utf-8') as inf:
            try:
                data = json.load(inf)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {f_name}: {e}")
                continue

            for g_info in data:
                if g_info['problem_id'] not in name_dict:
                    name_dict[g_info['problem_id']] = name_num
                    name_num += 1

    return name_dict


class graph(object):
    def __init__(self, node_num=0, label=None, name=None):
        self.node_num = node_num
        self.label = label
        self.name = name
        self.type = name  # (cpp/java)
        self.features = []
        self.succs = {'normal': [], 'cfg': [], 'dfg': []}  
        self.preds = {'normal': [], 'cfg': [], 'dfg': []}
        if node_num > 0:
            node_num += 4  
            for i in range(node_num):
                self.features.append([])
                for edge_type in self.succs:
                    self.succs[edge_type].append([])
                    self.preds[edge_type].append([])

    def add_node(self, feature=[]):
        self.node_num += 3
        self.features.append(feature)
        for edge_type in self.succs:
            self.succs[edge_type].append([])
            self.preds[edge_type].append([])

    def add_edge(self, u, v, edge_type='normal'):
        self.succs[edge_type][u].append(v)
        self.preds[edge_type][v].append(u)

    def toString(self):
        ret = '{} {}\n'.format(self.node_num, self.label)
        for u in range(self.node_num):
            for fea in self.features[u]:
                ret += '{} '.format(fea)
            ret += str(len(self.succs['normal'][u]))  
            for succ in self.succs['normal'][u]:
                ret += ' {}'.format(succ)
            ret += '\n'
        return ret

def read_graph(F_NAME, FUNC_NAME_DICT, FEATURE_DIM, n_num):
    graphs = []
    classes = []
    if FUNC_NAME_DICT is not None:
        for f in range(len(FUNC_NAME_DICT)):
            classes.append([])

    for f_name in F_NAME:
        with open(f_name) as inf:
            for line in inf:
                line = line.strip('[').strip().strip(',').strip(']')
                g_info = json.loads(line.strip())
                label = FUNC_NAME_DICT[g_info['problem_id']]
                classes[label].append(len(graphs))
                cur_graph = graph(g_info['n_num'], label, g_info['type'])

                # if g_info['type'] == 'cpp':
                #     n_java += 1
                if n_num < g_info['n_num']:
                    n_num = g_info['n_num']


                for u in range(g_info['n_num'] - 2):
                    cur_graph.features[u] = np.array(g_info['features'][u])


                    for v in g_info['Edges'][u]:
                        cur_graph.add_edge(u, v, edge_type='normal')


                    if 'cfgEdges' in g_info and u < len(g_info['cfgEdges']):
                        for v in g_info['cfgEdges'][u]:
                            cur_graph.add_edge(u, v, edge_type='cfg')


                    if 'ddgEdges' in g_info and u < len(g_info['ddgEdges']):
                        for v in g_info['ddgEdges'][u]:
                            cur_graph.add_edge(u, v, edge_type='dfg')

                graphs.append(cur_graph)
    return graphs, classes, n_num


def partition_data(Gs, classes, partitions, perm):
    C = len(classes)
    st = 0.0
    ret = []

    for part in partitions:
        cur_g = []
        cur_c = []
        ed = st + part * C

        for cls in range(int(st), int(ed)):
            prev_class = classes[perm[cls]]
            cur_c.append([])
            for i in range(len(prev_class)):
                cur_g.append(Gs[prev_class[i]])
                cur_g[-1].label = len(cur_c) - 1
                cur_c[-1].append(len(cur_g) - 1)

        ret.append(cur_g)
        ret.append(cur_c)
        st = ed
    return ret


def generate_epoch_pair(Gs, classes, M, output_id=False, load_id=None):
    epoch_data = []
    id_data = []  # [ ([(G0,G1),(G0,G1), ...], [(G0,H0),(G0,H0), ...]), ... ]

    if load_id is None:
        st = 0
        while st < len(Gs):
            if output_id:
                X1, X2, m1, m2, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2, y, pos_id, neg_id = get_pair(Gs, classes, M,
                                                                                                         st=st,
                                                                                                         output_id=True)
                id_data.append((pos_id, neg_id))
            else:
                X1, X2, m1, m2, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2, y = get_pair(Gs, classes, M, st=st)

            epoch_data.append((X1, X2, m1, m2, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2, y))
            st += M
    else:  # Load from previous id data
        id_data = load_id
        for id_pair in id_data:
            X1, X2, m1, m2, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2, y = get_pair(Gs, classes, M, load_id=id_pair)
            epoch_data.append((X1, X2, m1, m2, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2, y))

    if output_id:
        return epoch_data, id_data
    else:
        return epoch_data

def get_pair(Gs, classes, M, st=-1, output_id=False, load_id=None):
    if load_id is None:
        C = len(classes)

        if (st + M > len(Gs)):
            M = len(Gs) - st
        ed = st + M

        pos_ids = []  # [(G_0, G_1)]
        neg_ids = []  # [(G_0, H_0)]

        for g_id in range(st, ed):
            g0 = Gs[g_id]
            cls = g0.label

            tot_g = len(classes[cls])
            if (len(classes[cls]) >= 2):
                g1_id = classes[cls][np.random.randint(tot_g)]
                while g_id == g1_id and Gs[g_id].name == Gs[g1_id].name:
                    g1_id = classes[cls][np.random.randint(tot_g)]
                while g_id == g1_id:
                    g1_id = classes[cls][np.random.randint(tot_g)]
                pos_ids.append((g_id, g1_id))

            cls2 = np.random.randint(C)
            while (len(classes[cls2]) == 0) or (cls2 == cls):
                cls2 = np.random.randint(C)
            tot_g2 = len(classes[cls2])
            h_id = classes[cls2][np.random.randint(tot_g2)]
            while g_id == h_id and Gs[h_id].name == Gs[g_id].name:
                h_id = classes[cls2][np.random.randint(tot_g2)]
            while h_id == g_id:
                h_id = classes[cls2][np.random.randint(tot_g2)]
            neg_ids.append((g_id, h_id))
    else:
        pos_ids = load_id[0]
        neg_ids = load_id[1]

    M_pos = len(pos_ids)
    M_neg = len(neg_ids)
    M = M_pos + M_neg

    maxN1 = 0
    maxN2 = 0
    for pair in pos_ids:
        maxN1 = max(maxN1, Gs[pair[0]].node_num + 4)
        maxN2 = max(maxN2, Gs[pair[1]].node_num + 4)
    for pair in neg_ids:
        maxN1 = max(maxN1, Gs[pair[0]].node_num + 4)
        maxN2 = max(maxN2, Gs[pair[1]].node_num + 4)

    feature_dim = 65

    X1_input = np.zeros((M, maxN1, feature_dim))
    X2_input = np.zeros((M, maxN2, feature_dim))
    node1_mask = np.zeros((M, maxN1, maxN1))
    node2_mask = np.zeros((M, maxN2, maxN2))
    cfg_mask1 = np.zeros((M, maxN1, maxN1))
    cfg_mask2 = np.zeros((M, maxN2, maxN2))
    ddg_mask1 = np.zeros((M, maxN1, maxN1))
    ddg_mask2 = np.zeros((M, maxN2, maxN2))
    y_input = np.zeros((M))

    def adjust_features(feature, target_dim):
        if len(feature) > target_dim:
            return np.array(feature[:target_dim])
        else:
            return np.pad(feature, (0, target_dim - len(feature)), 'constant')

    for i in range(M_pos):
        y_input[i] = 1
        g1 = Gs[pos_ids[i][0]]
        g2 = Gs[pos_ids[i][1]]
        for u in range(g1.node_num - 4):
            X1_input[i, u, :] = adjust_features(g1.features[u], feature_dim)
            for v in g1.succs['normal'][u]:
                node1_mask[i, u, v] = 1
            for v in g1.succs['cfg'][u]:
                cfg_mask1[i, u, v] = 1
            for v in g1.succs['dfg'][u]:
                ddg_mask1[i, u, v] = 1

        for u in range(g2.node_num - 4):
            X2_input[i, u, :] = adjust_features(g2.features[u], feature_dim)
            for v in g2.succs['normal'][u]:
                node2_mask[i, u, v] = 1
            for v in g2.succs['cfg'][u]:
                cfg_mask2[i, u, v] = 1
            for v in g2.succs['dfg'][u]:
                ddg_mask2[i, u, v] = 1

    for i in range(M_pos, M_pos + M_neg):
        y_input[i] = -1
        g1 = Gs[neg_ids[i - M_pos][0]]
        g2 = Gs[neg_ids[i - M_pos][1]]
        for u in range(g1.node_num - 4):
            X1_input[i, u, :] = adjust_features(g1.features[u], feature_dim)
            for v in g1.succs['normal'][u]:
                node1_mask[i, u, v] = 1
            for v in g1.succs['cfg'][u]:
                cfg_mask1[i, u, v] = 1
            for v in g1.succs['dfg'][u]:
                ddg_mask1[i, u, v] = 1

        for u in range(g2.node_num - 4):
            X2_input[i, u, :] = adjust_features(g2.features[u], feature_dim)
            for v in g2.succs['normal'][u]:
                node2_mask[i, u, v] = 1
            for v in g2.succs['cfg'][u]:
                cfg_mask2[i, u, v] = 1
            for v in g2.succs['dfg'][u]:
                ddg_mask2[i, u, v] = 1

    if output_id:
        return X1_input, X2_input, node1_mask, node2_mask, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2, y_input, pos_ids, neg_ids
    else:
        return X1_input, X2_input, node1_mask, node2_mask, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2, y_input


def train_epoch(model, graphs, classes, batch_size, load_data=None):
    if load_data is None:
        epoch_data = generate_epoch_pair(graphs, classes, batch_size)
    else:
        epoch_data = load_data

    perm = np.random.permutation(len(epoch_data))

    cum_loss = 0.0
    for index in perm:
        cur_data = epoch_data[index]
        X1, X2, mask1, mask2, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2, y = cur_data

        # print(mask1.shape)
        # print(mask2.shape)

        loss = model.train(X1, X2, mask1, mask2, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2,y)
        cum_loss += loss

    return cum_loss / len(perm)
def get_auc_epoch(model, graphs, classes, batch_size, load_data=None):
    tot_diff = []
    tot_truth = []

    if load_data is None:
        epoch_data = generate_epoch_pair(graphs, classes, batch_size)
    else:
        epoch_data = load_data

    for cur_data in epoch_data:
        X1, X2, m1, m2, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2, y = cur_data

        diff = model.calc_diff(X1, X2, m1, m2, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2)

        tot_diff += list(diff)
        tot_truth += list(y > 0)

    diff = np.array(tot_diff)
    truth = np.array(tot_truth)
    fpr, tpr, thresholds = roc_curve(truth, (1 - diff) / 2)
    model_auc = auc(fpr, tpr)

    series = pd.Series((1 - diff) / 2)
    predicts = series.apply(lambda x: 1 if x >= 0.5 else 0)
    predicts.reset_index(drop=True, inplace=True)

    f1 = f1_score(truth, predicts)
    ac = accuracy_score(truth, predicts)
    re = recall_score(truth, predicts)
    pr = precision_score(truth, predicts)

    return model_auc, fpr, tpr, thresholds, f1, ac, re, pr



