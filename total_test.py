import tensorflow as tf

print(tf.__version__)
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from GATSiamese import graphnn1
from GNNSiamese import graphnn2
from GATSiamese_AST import graphnn3
from GNNSiamese_AST import graphnn4
from utilst import *
from utilst_AST import *
import os
import argparse
import json
from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0',
                    help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=65,
                    help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=128,
                    help='embedding dimension')
# parser.add_argument('--embed_dim_2', type=int, default=32,
#                     help='embedding dimension')
# parser.add_argument('--embed_dim_3', type=int, default=64,
#                     help='embedding dimension')
# parser.add_argument('--embed_dim_4', type=int, default=128,
#                     help='embedding dimension')
# parser.add_argument('--embed_dim_5', type=int, default=256,
#                     help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=5,
                    help='embedding network depth')
# parser.add_argument('--embed_depth_2', type=int, default=2,
#                     help='embedding network depth')
# parser.add_argument('--embed_depth_3', type=int, default=3,
#                     help='embedding network depth')
# parser.add_argument('--embed_depth_4', type=int, default=4,
#                     help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=128,
                    help='output layer dimension')
parser.add_argument('--iter_level_1', type=int, default=3,
                    help='iteration times')
parser.add_argument('--iter_level_2', type=int, default=3,
                    help='iteration times')
parser.add_argument('--iter_level_3', type=int, default=3,
                    help='iteration times')
parser.add_argument('--iter_level_4', type=int, default=3,
                    help='iteration times')
# parser.add_argument('--iter_level_5', type=int, default=5,
#                     help='iteration times')
# parser.add_argument('--iter_level_6', type=int, default=5,
#                     help='iteration times')
# parser.add_argument('--iter_level_7', type=int, default=5,
#                     help='iteration times')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--epoch', type=int, default=100,
                    help='epoch number')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--load_path_1', type=str,
                    default='models_AtCode/saved_models_gat_100',
                    help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--load_path_2', type=str,
                    default='models_AtCode/saved_model_gnn_100',
                    help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--load_path_3', type=str,
                    default='models_AtCode/saved_model_ast_gat_100',
                    help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--load_path_4', type=str,
                    default='models_AtCode/saved_mode_ast_gnn_100',
                    help='path for model loading, "#LATEST#" for the latest checkpoint')

parser.add_argument('--log_path', type=str, default=None,
                    help='path for training log')

if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print("=================================")
    print(args)
    print("=================================")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    Dtype = args.dtype
    NODE_FEATURE_DIM = args.fea_dim
    EMBED_DIM = args.embed_dim

    EMBED_DEPTH = args.embed_depth

    OUTPUT_DIM = args.output_dim
    ITERATION_LEVEL_1 = args.iter_level_1
    ITERATION_LEVEL_2 = args.iter_level_2
    ITERATION_LEVEL_3 = args.iter_level_3
    ITERATION_LEVEL_4 = args.iter_level_4
    LEARNING_RATE = args.lr
    MAX_EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    LOAD_PATH_1 = args.load_path_1
    LOAD_PATH_2 = args.load_path_2
    LOAD_PATH_3 = args.load_path_3
    LOAD_PATH_4 = args.load_path_4

    LOG_PATH = args.log_path

    SHOW_FREQ = 1
    TEST_FREQ = 1
    SAVE_FREQ = 5
    # DATA_FILE_NAME = './data/test2/c++&Java(8000).json'.format(NODE_FEATURE_DIM)
    # SOFTWARE=('openssl-1.0.1f-', 'openssl-1.0.1u-')
    # OPTIMIZATION=('-O0', '-O1','-O2','-O3')
    # COMPILER=('armeb-linux', 'i586-linux', 'mips-linux')
    # VERSION=('v54',)

    FUNC_NAME_DICT = {}
    n_num=0
    # Process the input graphs
    # F_NAME = get_f_name(DATA_FILE_NAME, SOFTWARE, COMPILER,
    # OPTIMIZATION, VERSION)
    F_NAME = get_f_name()
    FUNC_NAME_DICT = get_f_dict(F_NAME)

    Gs, classes ,n_num= read_graph(F_NAME, FUNC_NAME_DICT, NODE_FEATURE_DIM,n_num)
    print("{} graphs, {} functions".format(len(Gs), len(classes)))

    if os.path.isfile('data/class_perm.npy'):
        perm = np.load('data/class_perm.npy')
    else:
        perm = np.random.permutation(len(classes))
        np.save('data/class_perm.npy', perm)
    if len(perm) < len(classes):
        perm = np.random.permutation(len(classes))
        np.save('data/class_perm.npy', perm)

    Gs_train, classes_train, Gs_dev, classes_dev, Gs_test, classes_test = \
        partition_data(Gs, classes, [0.6, 0.2, 0.2], perm)


    print("Train: {} graphs, {} functions".format(
        len(Gs_train), len(classes_train)))
    print("Dev: {} graphs, {} functions".format(
        len(Gs_dev), len(classes_dev)))
    print("Test: {} graphs, {} functions".format(
        len(Gs_test), len(classes_test)))

    # Fix the pairs for validation and testing
    if os.path.isfile('data/valid2.json'):
        with open('data/valid2.json') as inf:
            valid_ids = json.load(inf)
        valid_epoch = generate_epoch_pair(
            Gs_dev, classes_dev, BATCH_SIZE, load_id=valid_ids)
    else:
        valid_epoch, valid_ids = generate_epoch_pair(
            Gs_dev, classes_dev, BATCH_SIZE, output_id=True)
        with open('data/valid2.json', 'w') as outf:
            json.dump(valid_ids, outf)

    if os.path.isfile('data/test.json'):
        with open('data/test.json') as inf:
            test_ids = json.load(inf)
        test_epoch = generate_epoch_pair(
            Gs_test, classes_test, BATCH_SIZE, load_id=test_ids)
    else:
        test_epoch, test_ids = generate_epoch_pair(
            Gs_test, classes_test, BATCH_SIZE, output_id=True)
        with open('data/test.json', 'w') as outf:
            json.dump(test_ids, outf)

    # Model1
    gnn = graphnn1(
        N_x=NODE_FEATURE_DIM,
        Dtype=Dtype,
        N_embed=EMBED_DIM,
        depth_embed=EMBED_DEPTH,
        N_o=OUTPUT_DIM,
        ITER_LEVEL=ITERATION_LEVEL_1,
        lr=LEARNING_RATE
    )
    gnn.init(LOAD_PATH_1, LOG_PATH)

    # Test
    val_auc, fpr1, tpr1, thres, f1, ac, re, pr = get_auc_epoch(
        gnn, Gs_dev, classes_dev, BATCH_SIZE, load_data=valid_epoch)
    gnn.say("AUC on validation set: {}".format(val_auc))
    test_auc, fpr1, tpr1, thres, f1, ac, re, pr = get_auc_epoch(
        gnn, Gs_test, classes_test, BATCH_SIZE, load_data=test_epoch)
    print("Accuracy1:", ac)
    print("precision:", pr)
    print("recall:", re)
    print("F1score:", f1)
    print("fpr:", fpr1)
    print("tpr:", tpr1)
    roc_auc1 = metrics.auc(fpr1, tpr1)
    gnn.say("AUC on testing set: {}".format(test_auc))
    
    # Model2
    gnn = graphnn2(
        N_x=NODE_FEATURE_DIM,
        Dtype=Dtype,
        N_embed=EMBED_DIM,
        depth_embed=EMBED_DEPTH,
        N_o=OUTPUT_DIM,
        ITER_LEVEL=ITERATION_LEVEL_2,
        lr=LEARNING_RATE
    )
    gnn.init(LOAD_PATH_2, LOG_PATH)

    # Test
    val_auc, fpr2, tpr2, thres, f1, ac, re, pr = get_auc_epoch(
        gnn, Gs_dev, classes_dev, BATCH_SIZE, load_data=valid_epoch)
    gnn.say("AUC on validation set: {}".format(val_auc))
    test_auc, fpr2, tpr2, thres, f1, ac, re, pr = get_auc_epoch(
        gnn, Gs_test, classes_test, BATCH_SIZE, load_data=test_epoch)
    print("Accuracy2:", ac)
    print("precision:", pr)
    print("recall:", re)
    print("F1score:", f1)
    print("fpr:", fpr2)
    print("tpr:", tpr2)
    roc_auc2 = metrics.auc(fpr2, tpr2)



    FUNC_NAME_DICT = {}
    n_num=0
    # Process the input graphs
    # F_NAME = get_f_name(DATA_FILE_NAME, SOFTWARE, COMPILER,
    # OPTIMIZATION, VERSION)
    F_NAME = get_f_name_AST()
    # FUNC_NAME_DICT = get_f_dict(F_NAME)
    FUNC_NAME_DICT = get_f_dict_AST(F_NAME)

    Gs, classes ,n_num= read_graph_AST(F_NAME, FUNC_NAME_DICT, NODE_FEATURE_DIM,n_num)
    print("{} graphs, {} functions".format(len(Gs), len(classes)))

    if os.path.isfile('data/class_perm.npy'):
        perm = np.load('data/class_perm.npy')
    else:
        perm = np.random.permutation(len(classes))
        np.save('data/class_perm.npy', perm)
    if len(perm) < len(classes):
        perm = np.random.permutation(len(classes))
        np.save('data/class_perm.npy', perm)

    Gs_train, classes_train, Gs_dev, classes_dev, Gs_test, classes_test = \
        partition_data_AST(Gs, classes, [0.6, 0.2, 0.2], perm)

    print("Train: {} graphs, {} functions".format(
        len(Gs_train), len(classes_train)))
    print("Dev: {} graphs, {} functions".format(
        len(Gs_dev), len(classes_dev)))
    print("Test: {} graphs, {} functions".format(
        len(Gs_test), len(classes_test)))

    # Fix the pairs for validation and testing
    if os.path.isfile('data/valid2.json'):
        with open('data/valid2.json') as inf:
            valid_ids = json.load(inf)
        valid_epoch = generate_epoch_pair_AST(
            Gs_dev, classes_dev, BATCH_SIZE, load_id=valid_ids)
    else:
        valid_epoch, valid_ids = generate_epoch_pair_AST(
            Gs_dev, classes_dev, BATCH_SIZE, output_id=True)
        with open('data/valid2.json', 'w') as outf:
            json.dump(valid_ids, outf)

    if os.path.isfile('data/test.json'):
        with open('data/test.json') as inf:
            test_ids = json.load(inf)
        test_epoch = generate_epoch_pair_AST(
            Gs_test, classes_test, BATCH_SIZE, load_id=test_ids)
    else:
        test_epoch, test_ids = generate_epoch_pair_AST(
            Gs_test, classes_test, BATCH_SIZE, output_id=True)
        with open('data/test.json', 'w') as outf:
            json.dump(test_ids, outf)
    # Model3
    gnn = graphnn3(
        N_x=NODE_FEATURE_DIM,
        Dtype=Dtype,
        N_embed=EMBED_DIM,
        depth_embed=EMBED_DEPTH,
        N_o=OUTPUT_DIM,
        ITER_LEVEL=ITERATION_LEVEL_3,
        lr=LEARNING_RATE
    )
    gnn.init(LOAD_PATH_3, LOG_PATH)

    # Test
    val_auc, fpr3, tpr3, thres, f1, ac, re, pr = get_auc_epoch_AST(
        gnn, Gs_dev, classes_dev, BATCH_SIZE, load_data=valid_epoch)
    gnn.say("AUC on validation set: {}".format(val_auc))
    test_auc, fpr3, tpr3, thres, f1, ac, re, pr = get_auc_epoch_AST(
        gnn, Gs_test, classes_test, BATCH_SIZE, load_data=test_epoch)
    print("Accuracy3:", ac)
    print("precision:", pr)
    print("recall:", re)
    print("F1score:", f1)
    print("fpr:", fpr3)
    print("tpr:", tpr3)
    roc_auc3 = metrics.auc(fpr3, tpr3)

    # Model
    gnn = graphnn4(
        N_x=NODE_FEATURE_DIM,
        Dtype=Dtype,
        N_embed=EMBED_DIM,
        depth_embed=EMBED_DEPTH,
        N_o=OUTPUT_DIM,
        ITER_LEVEL=ITERATION_LEVEL_4,
        lr=LEARNING_RATE
    )
    gnn.init(LOAD_PATH_4, LOG_PATH)

    # Test
    val_auc, fpr4, tpr4, thres, f1, ac, re, pr = get_auc_epoch_AST(
        gnn, Gs_dev, classes_dev, BATCH_SIZE, load_data=valid_epoch)
    gnn.say("AUC on validation set: {}".format(val_auc))
    test_auc, fpr4, tpr4, thres, f1, ac, re, pr = get_auc_epoch_AST(
        gnn, Gs_test, classes_test, BATCH_SIZE, load_data=test_epoch)
    print("Accuracy4:", ac)
    print("precision:", pr)
    print("recall:", re)
    print("F1score:", f1)
    print("fpr:", fpr2)
    print("tpr:", tpr2)
    roc_auc4 = metrics.auc(fpr4, tpr4)

    plt.figure(figsize=(6, 6))
    plt.title('Validation ROC')
    plt.plot(fpr1, tpr1, 'r', label='FE+GAT AUC = %0.2f' % roc_auc1)
    plt.plot(fpr2, tpr2, 'g', label='FE+GNN AUC = %0.2f' % roc_auc2)
    plt.plot(fpr3, tpr3, 'b', label='AST+GAT AUC = %0.2f' % roc_auc3)
    plt.plot(fpr4, tpr4, 'c', label='AST+GNN AUC = %0.2f' % roc_auc4)
    # plt.plot(fpr5, tpr5, 'm', label='T=5 AUC = %0.2f' % roc_auc5)
    # plt.plot(fpr6, tpr6, 'y', label='T=6 AUC = %0.2f' % roc_auc6)
    # plt.plot(fpr7, tpr7, 'k', label='T=7 AUC = %0.2f' % roc_auc7)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("filename.png")
    plt.show()
    plt.close()

    gnn.say("AUC on testing set: {}".format(test_auc))
