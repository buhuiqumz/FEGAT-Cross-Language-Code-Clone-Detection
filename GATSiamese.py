
import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.metrics import roc_auc_score





def graph_embed(X, msg_mask, cfg_mask, ddg_mask, N_x, N_embed, N_o, iter_level, Wnode, Wembed, W_output, b_output, Wattention_ast,Wattention_cfg,Wattention_ddg):

    node_val = tf.reshape(tf.matmul(tf.reshape(X, [-1, N_x]), Wnode),
                          [tf.shape(X)[0], -1, N_embed])

    cur_msg = tf.nn.relu(node_val)  # [batch, node_num, embed_dim]

    for t in range(iter_level):

        # attention_input_ast = multi_head_attention(cur_msg, Wattention_ast[t])  # [batch, node_num, embed_dim]
        attention_input_ast = tf.tensordot(cur_msg, Wattention_ast[t], axes=[[2], [0]])  # [batch, node_num, embed_dim]

        attention_output_ast = tf.matmul(msg_mask, attention_input_ast)  # [batch, node_num, embed_dim]


        attention_input_cfg = tf.tensordot(cur_msg, Wattention_cfg[t], axes=[[2], [0]])  # [batch, node_num, embed_dim]
        attention_output_cfg = tf.matmul(cfg_mask, attention_input_cfg)  # [batch, node_num, embed_dim]


        attention_input_dfg = tf.tensordot(cur_msg, Wattention_ddg[t], axes=[[2], [0]])  # [batch, node_num, embed_dim]
        attention_output_dfg = tf.matmul(ddg_mask, attention_input_dfg)  # [batch, node_num, embed_dim]




        combined_output = attention_output_ast+attention_output_cfg + attention_output_dfg


        tot_val_t = node_val + combined_output
        cur_msg = tf.nn.tanh(tot_val_t)



    g_embed = tf.reduce_sum(cur_msg, 1)  # [batch, embed_dim]

    output = tf.matmul(g_embed, W_output) + b_output

    return output



class graphnn1(object):
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

        tf.reset_default_graph()
        with tf.device(device):
            Wnode = tf.Variable(tf.truncated_normal(
                shape=[N_x, N_embed], stddev=0.1, dtype=Dtype))
            Wembed = []
            for i in range(depth_embed):
                Wembed.append(tf.Variable(tf.truncated_normal(
                    shape=[N_embed, N_embed], stddev=0.1, dtype=Dtype)))
            num_heads=4

            Wattention_ast = []
            for i in range(depth_embed):
                Wattention_ast.append(tf.Variable(tf.truncated_normal(
                    shape=[N_embed, N_embed], stddev=0.1, dtype=Dtype)))
                # Wattention_ast.append(tf.Variable(tf.truncated_normal(shape=[num_heads, N_embed // num_heads, N_embed // num_heads],stddev=0.1, dtype=Dtype)))
            Wattention_cfg = []
            for i in range(depth_embed):
                Wattention_cfg.append(tf.Variable(tf.truncated_normal(
                    shape=[N_embed, N_embed], stddev=0.1, dtype=Dtype)))

            Wattention_ddg = []
            for i in range(depth_embed):
                Wattention_ddg.append(tf.Variable(tf.truncated_normal(
                    shape=[N_embed, N_embed], stddev=0.1, dtype=Dtype)))

            W_output = tf.Variable(tf.truncated_normal(
                shape=[N_embed, N_o], stddev=0.1, dtype=Dtype))
            b_output = tf.Variable(tf.constant(0, shape=[N_o], dtype=Dtype))

            X1 = tf.placeholder(Dtype, [None, None, N_x])  # [B, N_node, N_x]
            msg1_mask = tf.placeholder(Dtype, [None, None, None])
            self.cfg_mask1 = tf.compat.v1.placeholder(Dtype, [None, None, None])
            self.ddg_mask1 = tf.compat.v1.placeholder(Dtype, [None, None, None])
            # [B, N_node, N_node]
            self.X1 = X1
            self.msg1_mask = msg1_mask
            embed1 = graph_embed(X1, msg1_mask, self.cfg_mask1, self.ddg_mask1, N_x, N_embed, N_o, ITER_LEVEL,
                                 Wnode, Wembed, W_output, b_output,Wattention_ast,Wattention_cfg,Wattention_ddg)  # [B, N_x]

            # embed1 = graph_embed(X1, msg1_mask, self.cfg_mask1, self.ddg_mask1, N_x, N_embed, N_o, ITER_LEVEL,
            #                      Wnode, Wembed, W_output, b_output,Wattention_ast)  # [B, N_x]

            X2 = tf.placeholder(Dtype, [None, None, N_x])
            msg2_mask = tf.placeholder(Dtype, [None, None, None])
            self.cfg_mask2 = tf.compat.v1.placeholder(Dtype, [None, None, None])
            self.ddg_mask2 = tf.compat.v1.placeholder(Dtype, [None, None, None])

            self.X2 = X2
            self.msg2_mask = msg2_mask
            embed2 = graph_embed(X2, msg2_mask, self.cfg_mask2, self.ddg_mask2, N_x, N_embed, N_o, ITER_LEVEL,
                                 Wnode, Wembed, W_output, b_output,Wattention_ast,Wattention_cfg,Wattention_ddg)
            # embed2 = graph_embed(X2, msg2_mask, self.cfg_mask2, self.ddg_mask2, N_x, N_embed, N_o, ITER_LEVEL,
            #                      Wnode, Wembed, W_output, b_output,Wattention_ast)

            label = tf.placeholder(Dtype, [None, ])  # same: 1; different:-1
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
        if self.log_file != None:
            self.log_file.write(string + '\n')

    def init(self, LOAD_PATH, LOG_PATH):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.Saver()
        self.sess = sess
        self.saver = saver
        self.log_file = None
        if (LOAD_PATH is not None):
            if LOAD_PATH == '#LATEST#':
                checkpoint_path = tf.train.latest_checkpoint('./')
            else:
                checkpoint_path = LOAD_PATH
            saver.restore(sess, checkpoint_path)
            if LOG_PATH != None:
                self.log_file = open(LOG_PATH, 'a+')
            self.say('{}, model loaded from file: {}'.format(
                datetime.datetime.now(), checkpoint_path))
        else:
            sess.run(tf.global_variables_initializer())
            if LOG_PATH != None:
                self.log_file = open(LOG_PATH, 'w')
            self.say('Training start @ {}'.format(datetime.datetime.now()))

    def get_embed(self, X1, mask1,cfg_mask1, ddg_mask1):
        vec, = self.sess.run(fetches=[self.embed1],
                             feed_dict={self.X1: X1, self.msg1_mask: mask1,self.cfg_mask1: cfg_mask1, self.ddg_mask1: ddg_mask1})
        return vec

    def calc_loss(self, X1, X2, mask1, mask2, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2, y):
        cur_loss, = self.sess.run(fetches=[self.loss], feed_dict={self.X1: X1,
                                                                  self.X2: X2, self.msg1_mask: mask1,
                                                                  self.msg2_mask: mask2,
                                                                  self.cfg_mask1: cfg_mask1,
                                                                  self.cfg_mask2: cfg_mask2,
                                                                  self.ddg_mask1: ddg_mask1,
                                                                  self.ddg_mask2: ddg_mask2,
                                                                  self.label: y,})
        return cur_loss

    def calc_diff(self, X1, X2, mask1, mask2, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2):
        diff, = self.sess.run(fetches=[self.diff], feed_dict={self.X1: X1,
                                                              self.X2: X2, self.msg1_mask: mask1,
                                                              self.msg2_mask: mask2,
                                                              self.cfg_mask1: cfg_mask1,
                                                              self.cfg_mask2: cfg_mask2,
                                                              self.ddg_mask1: ddg_mask1,
                                                              self.ddg_mask2: ddg_mask2})
        return diff

    def train(self, X1, X2, mask1, mask2, cfg_mask1, cfg_mask2, ddg_mask1, ddg_mask2, y):
        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={self.X1: X1,
                                                                        self.X2: X2, self.msg1_mask: mask1,
                                                                        self.msg2_mask: mask2,
                                                                        self.cfg_mask1: cfg_mask1,
                                                                        self.cfg_mask2: cfg_mask2,
                                                                        self.ddg_mask1: ddg_mask1,
                                                                        self.ddg_mask2: ddg_mask2,
                                                                        self.label: y})
        return loss

    def save(self, path, epoch=None):
        checkpoint_path = self.saver.save(self.sess, path, global_step=epoch)
        return checkpoint_path


