import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.metrics import roc_auc_score


tf.compat.v1.disable_eager_execution()



def graph_embed(X, msg_mask, cfg_mask, ddg_mask, N_x, N_embed, N_o, iter_level, Wnode, Wembed, W_output, b_output):
    # X -- affine(W1) -- ReLU -- (Message -- affine(W2) -- add (with aff W1)
    # -- ReLU -- )* MessageAll  --  output
    node_val = tf.reshape(tf.matmul(tf.reshape(X, [-1, N_x]), Wnode),
                          [tf.shape(X)[0], -1, N_embed])

    cur_msg = tf.nn.relu(node_val)  # [batch, node_num, embed_dim]
    for t in range(iter_level):
        # Message convey
        Li_t = tf.matmul(msg_mask, cur_msg)  # [batch, node_num, embed_dim]

        cfg_msg = tf.matmul(cfg_mask, cur_msg)  # [batch, node_num, embed_dim]
        ddg_msg = tf.matmul(ddg_mask, cur_msg)  # [batch, node_num, embed_dim]

        combined_msg = Li_t + cfg_msg + ddg_msg  # 将所有消息结合

        # Complex Function
        cur_info = tf.reshape(combined_msg, [-1, N_embed])
        for Wi in Wembed:
            if Wi == Wembed[-1]:
                cur_info = tf.matmul(cur_info, Wi)
            else:
                cur_info = tf.nn.relu(tf.matmul(cur_info, Wi))

        neigh_val_t = tf.reshape(cur_info, tf.shape(Li_t))
        # Adding
        tot_val_t = node_val + neigh_val_t
        # Nonlinearity
        tot_msg_t = tf.nn.tanh(tot_val_t)
        cur_msg = tot_msg_t  # [batch, node_num, embed_dim]

    g_embed = tf.reduce_sum(cur_msg, 1)  # [batch, embed_dim]
    output = tf.matmul(g_embed, W_output) + b_output

    return output


class graphnn2(object):
    def __init__(self,
                 N_x,
                 Dtype,
                 N_embed,
                 depth_embed,
                 N_o,
                 ITER_LEVEL,
                 lr,
                 device='/gpu:0'
                 ):
        self.NODE_LABEL_DIM = N_x
        tf.compat.v1.reset_default_graph()
        with tf.device(device):
            Wnode = tf.Variable(tf.random.truncated_normal(
                shape=[N_x, N_embed], stddev=0.1, dtype=Dtype))
            Wembed = [tf.Variable(tf.random.truncated_normal(
                shape=[N_embed, N_embed], stddev=0.1, dtype=Dtype)) for _ in range(depth_embed)]

            W_output = tf.Variable(tf.random.truncated_normal(
                shape=[N_embed, N_o], stddev=0.1, dtype=Dtype))
            b_output = tf.Variable(tf.constant(0, shape=[N_o], dtype=Dtype))

            # Input placeholders
            X1 = tf.compat.v1.placeholder(Dtype, [None, None, N_x])  # [B, N_node, N_x]
            msg1_mask = tf.compat.v1.placeholder(Dtype, [None, None, None])
            self.cfg_mask1 = tf.compat.v1.placeholder(Dtype, [None, None, None])
            self.ddg_mask1 = tf.compat.v1.placeholder(Dtype, [None, None, None])

            self.X1 = X1
            self.msg1_mask = msg1_mask
            embed1 = graph_embed(X1, msg1_mask, self.cfg_mask1, self.ddg_mask1, N_x, N_embed, N_o, ITER_LEVEL,
                                 Wnode, Wembed, W_output, b_output)

            X2 = tf.compat.v1.placeholder(Dtype, [None, None, N_x])
            msg2_mask = tf.compat.v1.placeholder(Dtype, [None, None, None])
            self.cfg_mask2 = tf.compat.v1.placeholder(Dtype, [None, None, None])
            self.ddg_mask2 = tf.compat.v1.placeholder(Dtype, [None, None, None])

            self.X2 = X2
            self.msg2_mask = msg2_mask
            embed2 = graph_embed(X2, msg2_mask, self.cfg_mask2, self.ddg_mask2, N_x, N_embed, N_o, ITER_LEVEL,
                                 Wnode, Wembed, W_output, b_output)

            label = tf.compat.v1.placeholder(Dtype, [None, ])  # same: 1; different:-1
            self.label = label
            self.embed1 = embed1

            cos = tf.reduce_sum(embed1 * embed2, 1) / tf.sqrt(tf.reduce_sum(
                embed1 ** 2, 1) * tf.reduce_sum(embed2 ** 2, 1) + 1e-10)

            diff = -cos
            self.diff = diff
            loss = tf.reduce_mean((diff + label) ** 2)
            self.loss = loss

            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
            self.optimizer = optimizer

    def say(self, string):
        print(string)
        if self.log_file is not None:
            self.log_file.write(string + '\n')

    def init(self, LOAD_PATH, LOG_PATH):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.Saver()
        self.sess = sess
        self.saver = saver
        self.log_file = None
        if LOAD_PATH is not None:
            if LOAD_PATH == '#LATEST#':
                checkpoint_path = tf.train.latest_checkpoint('./')
            else:
                checkpoint_path = LOAD_PATH
            saver.restore(sess, checkpoint_path)
            if LOG_PATH is not None:
                self.log_file = open(LOG_PATH, 'a+')
            self.say('{}, model loaded from file: {}'.format(datetime.datetime.now(), checkpoint_path))
        else:
            sess.run(tf.global_variables_initializer())
            if LOG_PATH is not None:
                self.log_file = open(LOG_PATH, 'w')
            self.say('Training start @ {}'.format(datetime.datetime.now()))

    def get_embed(self, X1, mask1, cfg_mask1, ddg_mask1):
        vec, = self.sess.run(fetches=[self.embed1],
                             feed_dict={self.X1: X1, self.msg1_mask: mask1,
                                        self.cfg_mask1: cfg_mask1, self.ddg_mask1: ddg_mask1})
        return vec

    def calc_loss(self, X1, X2, mask1, mask2, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2, y):
        cur_loss, = self.sess.run(fetches=[self.loss], feed_dict={self.X1: X1,
                                                                  self.X2: X2,
                                                                  self.msg1_mask: mask1,
                                                                  self.msg2_mask: mask2,
                                                                  self.cfg_mask1: cfg_mask1,
                                                                  self.cfg_mask2: cfg_mask2,
                                                                  self.ddg_mask1: ddg_mask1,
                                                                  self.ddg_mask2: ddg_mask2,
                                                                  self.label: y})
        return cur_loss

    def calc_diff(self, X1, X2, mask1, mask2, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2):
        diff, = self.sess.run(fetches=[self.diff], feed_dict={self.X1: X1,
                                                              self.X2: X2,
                                                              self.msg1_mask: mask1,
                                                              self.msg2_mask: mask2,
                                                              self.cfg_mask1: cfg_mask1,
                                                              self.cfg_mask2: cfg_mask2,
                                                              self.ddg_mask1: ddg_mask1,
                                                              self.ddg_mask2: ddg_mask2})
        return diff

    def train(self, X1, X2, mask1, mask2, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2, y):
        _, cur_loss = self.sess.run(fetches=[self.optimizer, self.loss], feed_dict={self.X1: X1,
                                                                                     self.X2: X2,
                                                                                     self.msg1_mask: mask1,
                                                                                     self.msg2_mask: mask2,
                                                                                     self.cfg_mask1: cfg_mask1,
                                                                                     self.cfg_mask2: cfg_mask2,
                                                                                     self.ddg_mask1: ddg_mask1,
                                                                                     self.ddg_mask2: ddg_mask2,
                                                                                     self.label: y})
        return cur_loss

    def save(self, SAVE_PATH):
        self.saver.save(self.sess, SAVE_PATH)
        self.say('model saved @ {}'.format(datetime.datetime.now()))

    def close(self):
        if self.log_file is not None:
            self.log_file.close()
        self.sess.close()


    def calRes(logits, y):
        res = np.zeros(len(y))
        for i in range(len(y)):
            res[i] = 0 if logits[i] >= 0 else 1
        auc = roc_auc_score(y, logits)
        return res, auc

