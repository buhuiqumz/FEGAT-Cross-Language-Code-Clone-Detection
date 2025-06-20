import tensorflow as tf
print(tf.__version__)
#import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from GATSiamese import graphnn1
from utilst import *
import os
import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0',
        help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=65,
        help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=128,
        help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=5,
        help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=128,
        help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=3,
        help='iteration times')
parser.add_argument('--lr', type=float, default=1e-3,
        help='learning rate')
parser.add_argument('--epoch', type=int, default=100,
        help='epoch number')
parser.add_argument('--batch_size', type=int, default=32,
        help='batch size')
parser.add_argument('--load_path', type=str, default=None,
        help='path for model loading, "#LATEST#" for the latest checkpoint')
parser.add_argument('--save_path', type=str,
        default='./saved_model/graphnn-model', help='path for model saving')
parser.add_argument('--log_path', type=str, default=None,
        help='path for training log')




if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print("=================================")
    print(args)
    print("=================================")

    #os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    Dtype = args.dtype
    NODE_FEATURE_DIM = args.fea_dim
    EMBED_DIM = args.embed_dim
    EMBED_DEPTH = args.embed_depth
    OUTPUT_DIM = args.output_dim
    ITERATION_LEVEL = args.iter_level
    LEARNING_RATE = args.lr
    MAX_EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    LOAD_PATH = args.load_path
    SAVE_PATH = args.save_path
    LOG_PATH = args.log_path

    SHOW_FREQ = 1
    TEST_FREQ = 1
    SAVE_FREQ = 5
    n_num = 0
    n_java = 0
    #SOFTWARE=('openssl-1.0.1f-', 'openssl-1.0.1u-')
    #OPTIMIZATION=('-O0', '-O1','-O2','-O3')
    #COMPILER=('armeb-linux', 'i586-linux', 'mips-linux')
    #VERSION=('v54',)

    FUNC_NAME_DICT = {}

    # Process the input graphs
    #F_NAME = get_f_name(DATA_FILE_NAME, SOFTWARE, COMPILER,
            #OPTIMIZATION, VERSION)
    F_NAME = get_f_name()
    #FUNC_NAME_DICT = get_f_dict(F_NAME)
    FUNC_NAME_DICT=get_f_dict(F_NAME)


    Gs, classes ,n_num,n_java= read_graph(F_NAME, FUNC_NAME_DICT, NODE_FEATURE_DIM,n_num,n_java)#Gs=graphs
    print("{} graphs, {} functions".format(len(Gs), len(classes)))
    print("n_num=",n_num)
    print("n_java=", n_java)
    if os.path.isfile('data/class_perm.npy'):
        perm = np.load('data/class_perm.npy')
    else:
        perm = np.random.permutation(len(classes))
        np.save('data/class_perm.npy', perm)
    if len(perm) < len(classes):
        perm = np.random.permutation(len(classes))
        np.save('data/class_perm.npy', perm)

    Gs_train, classes_train, Gs_dev, classes_dev, Gs_test, classes_test =\
            partition_data(Gs,classes,[0.6,0.2,0.2],perm)

    print("Train: {} graphs, {} functions".format(
            len(Gs_train), len(classes_train)))
    print ("Dev: {} graphs, {} functions".format(
            len(Gs_dev), len(classes_dev)))
    print ("Test: {} graphs, {} functions".format(
            len(Gs_test), len(classes_test)))
    # Gs_train = Gs_test

    # classes_train = classes_test

    # Fix the pairs for validation
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


    # Model
    gnn = graphnn1(
            N_x = NODE_FEATURE_DIM,
            Dtype = Dtype,
            N_embed = EMBED_DIM,
            depth_embed = EMBED_DEPTH,
            N_o = OUTPUT_DIM,
            ITER_LEVEL = ITERATION_LEVEL,
            lr = LEARNING_RATE
        )
    gnn.init(LOAD_PATH, LOG_PATH)

    # Train
    auc, fpr, tpr, thres,f1,ac,re,pr = get_auc_epoch(gnn, Gs_train, classes_train,
            BATCH_SIZE, load_data=valid_epoch)

    gnn.say("Initial training auc = {0} @ {1}".format(auc, datetime.now()))
    #print("Accuracy:",ac)
    #print("precision:", pr)
    #print("recall:",re)
    #print("F1score:",f1)
    auc0, fpr, tpr, thres,f1,ac,re,pr= get_auc_epoch(gnn, Gs_dev, classes_dev,
            BATCH_SIZE, load_data=valid_epoch)
    gnn.say("Initial validation auc = {0} @ {1}".format(auc0, datetime.now()))



    best_auc = 0
    for i in range(1, MAX_EPOCH+1):
        current_epoch = i + 1
        decay_epochs = 10
        learning_rate = LEARNING_RATE / (1 + current_epoch / decay_epochs)
        #gnn.set_lr(learning_rate)
        #print(learning_rate)
        l = train_epoch(gnn, Gs_train, classes_train, BATCH_SIZE)
        gnn.say("EPOCH {3}/{0}, loss = {1} @ {2}".format(
            MAX_EPOCH, l, datetime.now(), i))

        if (i % TEST_FREQ == 0):
            auc, fpr, tpr, thres,f1,ac,re,pr = get_auc_epoch(gnn, Gs_train, classes_train,
                    BATCH_SIZE, load_data=valid_epoch)
            gnn.say("Testing model: training auc = {0} @ {1}".format(
                auc, datetime.now()))

            auc, fpr, tpr, thres ,f1,ac,re,pr= get_auc_epoch(gnn, Gs_dev, classes_dev,
                    BATCH_SIZE, load_data=valid_epoch)
            gnn.say("Testing model: validation auc = {0} @ {1}".format(
                auc, datetime.now()))

            if auc > best_auc:
                path = gnn.save(SAVE_PATH+'_best')
                best_auc = auc
                gnn.say("Model saved in {}".format(path))

        if (i % SAVE_FREQ == 0):
            path = gnn.save(SAVE_PATH, i)
            gnn.say("Model saved in {}".format(path))

