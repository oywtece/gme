import numpy as np
import tensorflow as tf
import datetime
import ctr_funcs as func
import config_gme as cfg
from time import time
import os
import shutil

# config
test_batch_size = cfg.test_batch_size
meta_mode = cfg.meta_mode
alpha = cfg.alpha
gamma = cfg.gamma
rnd_seed = cfg.rnd_seed

# must be from small to large
tar_idx = cfg.tar_idx
attr_idx = cfg.attr_idx

str_txt = cfg.output_file_name
base_path = './tmp'
model_loading_addr = cfg.model_loading_addr
model_saving_addr = base_path + '/meta_' + str_txt + '/'
output_file_name = base_path + '/meta_' + str_txt + '.txt'
save_model_ind = cfg.save_model_ind
num_csv_col_w_ngb = cfg.num_csv_col_w_ngb
train_file_name_a = cfg.train_file_name_a
train_file_name_b = cfg.train_file_name_b
test_file_name = cfg.test_file_name
batch_size = cfg.batch_size
n_ft = cfg.n_ft
k = cfg.k
kp_prob = cfg.kp_prob
n_epoch_meta = cfg.n_epoch_meta
record_step_size = cfg.record_step_size
layer_dim = cfg.layer_dim
att_dim = cfg.att_dim
opt_alg = cfg.opt_alg
n_one_hot_slot = cfg.n_one_hot_slot
n_mul_hot_slot = cfg.n_mul_hot_slot
max_len_per_slot = cfg.max_len_per_slot
input_format = cfg.input_format
n_slot = n_one_hot_slot + n_mul_hot_slot

n_one_hot_slot_ngb = cfg.n_one_hot_slot_ngb
n_mul_hot_slot_ngb = cfg.n_mul_hot_slot_ngb
max_len_per_slot_ngb = cfg.max_len_per_slot_ngb
max_n_ngb_ori = cfg.max_n_ngb_ori
max_n_ngb = cfg.max_n_ngb
n_slot_ngb = n_one_hot_slot_ngb + n_mul_hot_slot_ngb

meta_batch_size_range = cfg.meta_batch_size_range
cold_eta_range = cfg.cold_eta_range
meta_eta_range = cfg.meta_eta_range

label_col_idx = 0
record_defaults = [[0]]*num_csv_col_w_ngb
record_defaults[0] = [0.0]
total_num_ft_col = num_csv_col_w_ngb - 1

# key: slot_idx in ori data, val: col_idx in pred_emb
tar_slot_map = {}
for i in range(len(tar_idx)):
    tar_slot_map[tar_idx[i]] = i

## create para list
para_list = []
for ii in range(len(meta_batch_size_range)):
    for iii in range(len(cold_eta_range)):
        for iv in range(len(meta_eta_range)):
            para_list.append([meta_batch_size_range[ii], cold_eta_range[iii], \
                            meta_eta_range[iv]])

## record results
result_list = []

# loop over para_list
for item in para_list:
    meta_batch_size = item[0]
    cold_eta = item[1]
    meta_eta = item[2]

    tf.reset_default_graph()

    # create dir
    if not os.path.exists(base_path):
        os.mkdir(base_path)

#    # remove dir
#    if os.path.isdir(model_saving_addr):
#        shutil.rmtree(model_saving_addr)

    ###########################################################
    ###########################################################

    print('Loading data start!')
    tf.set_random_seed(rnd_seed)

    if input_format == 'tfrecord':
        # do not shuffle meta training data [already arranged]
        train_ft_a, train_label_a = func.tfrecord_input_pipeline_test(train_file_name_a, num_csv_col_w_ngb, meta_batch_size, n_epoch_meta)
        train_ft_b, train_label_b = func.tfrecord_input_pipeline_test(train_file_name_b, num_csv_col_w_ngb, meta_batch_size, n_epoch_meta)
        
        # need to load the same test file multiple times; once for each test
        test_ft, test_label = func.tfrecord_input_pipeline_test(test_file_name, num_csv_col_w_ngb, test_batch_size, 1)
        test_ft_meta, test_label_meta = func.tfrecord_input_pipeline_test(test_file_name, num_csv_col_w_ngb, test_batch_size, 1)
    print('Loading data done!')

    ########################################################################
    ########################################################################
    # data format (label is removed from x_input)
    # tar, ngb (w diff n_fts)
    def partition_input(x_input):
        # generate idx_list
        len_list = []
        # tar
        len_list.append(n_one_hot_slot)
        len_list.append(n_mul_hot_slot*max_len_per_slot)

        # ngb
        for _ in range(max_n_ngb_ori):
            len_list.append(n_one_hot_slot_ngb)
            len_list.append(n_mul_hot_slot_ngb*max_len_per_slot_ngb)

        len_list = np.array(len_list)
        idx_list = np.cumsum(len_list)

        x_input_one_hot = x_input[:, 0:idx_list[0]]
        x_input_mul_hot = x_input[:, idx_list[0]:idx_list[1]]
        # shape=[None, n_mul_hot_slot, max_len_per_slot]
        x_input_mul_hot = tf.reshape(x_input_mul_hot, [-1, n_mul_hot_slot, max_len_per_slot])

        #######################
        # ngb
        concat_one_hot_ngb = x_input[:, idx_list[1]:idx_list[2]]
        concat_mul_hot_ngb = x_input[:, idx_list[2]:idx_list[3]]
        for i in range(1, max_n_ngb_ori):
            # one_hot
            temp_1 = x_input[:, idx_list[2*i+1]:idx_list[2*i+2]]
            concat_one_hot_ngb = tf.concat([concat_one_hot_ngb, temp_1], 1)

            # mul_hot
            temp_2 = x_input[:, idx_list[2*i+2]:idx_list[2*i+3]]
            concat_mul_hot_ngb = tf.concat([concat_mul_hot_ngb, temp_2], 1)

        # shape=[None, max_n_ngb, n_one_hot_slot_ngb]
        concat_one_hot_ngb = tf.reshape(concat_one_hot_ngb, [-1, max_n_ngb_ori, n_one_hot_slot_ngb])

        # shape=[None, max_n_ngb, n_mul_hot_slot_ngb, max_len_per_slot_ngb]
        concat_mul_hot_ngb = tf.reshape(concat_mul_hot_ngb, [-1, max_n_ngb_ori, n_mul_hot_slot_ngb, \
                max_len_per_slot_ngb])

        x_input_one_hot_ngb = concat_one_hot_ngb[:, 0:max_n_ngb, :]
        x_input_mul_hot_ngb = concat_mul_hot_ngb[:, 0:max_n_ngb, :, :]

        return x_input_one_hot, x_input_mul_hot, x_input_one_hot_ngb, x_input_mul_hot_ngb

    # add mask
    def get_masked_one_hot(x_input_one_hot):
        data_mask = tf.cast(tf.greater(x_input_one_hot, 0), tf.float32)
        data_mask = tf.expand_dims(data_mask, axis = 2)
        data_mask = tf.tile(data_mask, (1,1,k))
        # output: (?, n_one_hot_slot, k)
        data_embed_one_hot = tf.nn.embedding_lookup(emb_mat, x_input_one_hot)
        data_embed_one_hot_masked = tf.multiply(data_embed_one_hot, data_mask)
        return data_embed_one_hot_masked

    def get_masked_mul_hot(x_input_mul_hot):
        data_mask = tf.cast(tf.greater(x_input_mul_hot, 0), tf.float32)
        data_mask = tf.expand_dims(data_mask, axis = 3)
        data_mask = tf.tile(data_mask, (1,1,1,k))
        # output: (?, n_mul_hot_slot, max_len_per_slot, k)
        data_embed_mul_hot = tf.nn.embedding_lookup(emb_mat, x_input_mul_hot)
        data_embed_mul_hot_masked = tf.multiply(data_embed_mul_hot, data_mask)
        # move reduce_sum here
        data_embed_mul_hot_masked = tf.reduce_sum(data_embed_mul_hot_masked, 2)
        return data_embed_mul_hot_masked

    # output: (?, n_one_hot_slot + n_mul_hot_slot, k)
    def get_concate_embed(x_input_one_hot, x_input_mul_hot):
        data_embed_one_hot = get_masked_one_hot(x_input_one_hot)
        data_embed_mul_hot = get_masked_mul_hot(x_input_mul_hot)
        data_embed_concat = tf.concat([data_embed_one_hot, data_embed_mul_hot], 1)
        return data_embed_concat

    def get_masked_one_hot_ngb(x_input_one_hot_ngb):
        data_mask = tf.cast(tf.greater(x_input_one_hot_ngb, 0), tf.float32)
        data_mask = tf.expand_dims(data_mask, axis = 3)
        data_mask = tf.tile(data_mask, (1,1,1,k))
        # output: (?, max_n_clk, n_one_hot_slot, k)
        data_embed_one_hot = tf.nn.embedding_lookup(emb_mat, x_input_one_hot_ngb)
        data_embed_one_hot_masked = tf.multiply(data_embed_one_hot, data_mask)
        return data_embed_one_hot_masked

    def get_masked_mul_hot_ngb(x_input_mul_hot_ngb):
        data_mask = tf.cast(tf.greater(x_input_mul_hot_ngb, 0), tf.float32)
        data_mask = tf.expand_dims(data_mask, axis = 4)
        data_mask = tf.tile(data_mask, (1,1,1,1,k))
        # output: (?, max_n_clk, n_mul_hot_slot, max_len_per_slot, k)
        data_embed_mul_hot = tf.nn.embedding_lookup(emb_mat, x_input_mul_hot_ngb)
        data_embed_mul_hot_masked = tf.multiply(data_embed_mul_hot, data_mask)
        # output: (?, max_n_clk, n_mul_hot_slot, k)
        data_embed_mul_hot_masked = tf.reduce_sum(data_embed_mul_hot_masked, 3)
        return data_embed_mul_hot_masked

    # output: (?, max_n_ngb, n_slot, k)
    def get_concate_embed_ngb(x_input_one_hot_ngb, x_input_mul_hot_ngb):
        data_embed_one_hot = get_masked_one_hot_ngb(x_input_one_hot_ngb)
        data_embed_mul_hot = get_masked_mul_hot_ngb(x_input_mul_hot_ngb)
        data_embed_concat = tf.concat([data_embed_one_hot, data_embed_mul_hot], 2)
        return data_embed_concat

    # col_idx starts from 0, wrt data_embed_concat
    def get_sel_col(data_embed_concat, col_idx):
        cur_col_idx = col_idx[0]
        # none * len(col_idx) * k
        ft_emb = data_embed_concat[:, cur_col_idx:cur_col_idx+1, :]
        for i in range(1, len(col_idx)):
            cur_col_idx = col_idx[i]
            cur_x = data_embed_concat[:, cur_col_idx:cur_col_idx+1, :]
            ft_emb = tf.concat([ft_emb, cur_x], 1)
        # reshape -> 2D none * total_dim
        ft_emb = tf.reshape(ft_emb, [-1, len(col_idx)*k])
        return ft_emb

    def get_sel_col_ngb(data_embed_concat_ngb, col_idx):
        cur_col_idx = col_idx[0]
        # none * max_n_ngb * len(col_idx) * k
        ngb_emb = data_embed_concat_ngb[:, :, cur_col_idx:cur_col_idx+1, :]
        for i in range(1, len(col_idx)):
            cur_col_idx = col_idx[i]
            cur_x = data_embed_concat_ngb[:, :, cur_col_idx:cur_col_idx+1, :]
            ngb_emb = tf.concat([ngb_emb, cur_x], 2)
        # reshape -> 3D none * max_n_ngb * total_dim
        ngb_emb = tf.reshape(ngb_emb, [-1, max_n_ngb, len(col_idx)*k])
        return ngb_emb

    # count number of valid (i.e., not padded with all 0) ngbs
    # output: none*1
    def count_n_valid_ngb(x_input_one_hot_ngb):
        # none * max_n_ngb * n_one_hot_slot_ngb
        data_mask_a = tf.cast(tf.greater(x_input_one_hot_ngb, 0), tf.float32)
        # none * max_n_ngb
        data_mask_a_reduce_sum = tf.reduce_sum(data_mask_a, 2)
        data_mask_b = tf.cast(tf.greater(data_mask_a_reduce_sum, 0), tf.float32)
        # none * 1
        n_valid = 1.0*tf.reduce_sum(data_mask_b, 1, keep_dims=True)
        return n_valid

    def gen_emb_from_self(data_embed_concat):
        tf.stop_gradient(emb_mat)
        ######### self attr #########
        # none * (len(attr_idx)*k)
        attr_emb = get_sel_col(data_embed_concat, attr_idx)
        # none * (len(tar_idx)*k)
        pred_emb_self = gamma * tf.nn.tanh(tf.matmul(attr_emb, W_meta))
        pred_emb_self = tf.reshape(pred_emb_self, [-1,len(tar_idx),k])
        return pred_emb_self

    def get_emb_from_ngb(data_embed_concat_ngb, x_input_one_hot_ngb):
        tf.stop_gradient(emb_mat)
        # none * max_n_ngb * (len(tar_idx) * k)
        ngb_emb = get_sel_col_ngb(data_embed_concat_ngb, tar_idx)
        n_valid_ngb = count_n_valid_ngb(x_input_one_hot_ngb)
        # must flatten first, otherwise [*,a,b] / [*,c] will result in err
        avg_ngb_emb = tf.layers.flatten(tf.reduce_sum(ngb_emb, 1)) / (n_valid_ngb + 1e-5)
        pred_emb = gamma * tf.nn.tanh(tf.matmul(avg_ngb_emb, W_meta))
        pred_emb = tf.reshape(pred_emb, [-1,len(tar_idx),k])
        return pred_emb

    # GME-P
    # src_idx_ngb - e.g., cre_id in ngb; ft_idx_ngb - other fts used to compute the similarity
    def gen_emb_from_self_and_ngb_pre(data_embed_concat, data_embed_concat_ngb):
        tf.stop_gradient(emb_mat)
        ######### self attr #########
        # none * (len(attr_idx)*k)
        attr_emb = get_sel_col(data_embed_concat, attr_idx)
        # none * (len(tar_idx)*k)
        pred_emb_self = gamma * tf.nn.tanh(tf.matmul(attr_emb, W_meta))

        # none * 1 * (len(tar_idx)*k)
        pred_emb_self_exp = tf.expand_dims(pred_emb_self, 1)
        # none * (max_n_ngb+1) * (len(tar_idx)*k)
        pred_emb_self_tile = tf.tile(pred_emb_self_exp, [1, max_n_ngb+1, 1])
        # convert to 2D [fold the first 2 dims]
        # (none*(max_n_ngb+1)) * (len(tar_idx)*k)
        pred_emb_self_2d = tf.reshape(pred_emb_self_tile, [-1, len(tar_idx)*k])

        ######### ngb #########
        # none * max_n_ngb * (len(tar_idx) * k)
        tar_emb_ngb_ori = get_sel_col_ngb(data_embed_concat_ngb, tar_idx)
        # none * 1 * (len(attr_idx)*k)
        pred_emb_exp = tf.expand_dims(pred_emb_self, 1)
        # none * (max_n_ngb + 1) * (len(attr_idx)*k)
        tar_emb_ngb = tf.concat([tar_emb_ngb_ori, pred_emb_exp], 1)
        # convert to 2D [fold the first 2 dims]
        tar_emb_ngb_2d = tf.reshape(tar_emb_ngb, [-1, len(tar_idx)*k])

        ######### GAT #########
        # (none*(max_ngb+1)) * (len(tar_idx)*k)
        temp_self = tf.matmul(pred_emb_self_2d, W_gat)
        temp_ngb = tf.matmul(tar_emb_ngb_2d, W_gat)
        # (none*(max_ngb+1)) * 1
        wgt = tf.nn.leaky_relu(tf.matmul(tf.concat([temp_self, temp_ngb], 1), a_gat))
        wgt = tf.reshape(wgt, [-1, max_n_ngb+1, 1])
        nlz_wgt = tf.nn.softmax(wgt, dim=1)

        temp_ngb_re = tf.reshape(temp_ngb, [-1, max_n_ngb+1, len(tar_idx)*k])
        # none * (len(tar_idx)*k)
        pred_emb_self_new = tf.nn.elu(tf.reduce_sum(temp_ngb_re * nlz_wgt, 1))
        # none * len(tar_idx) * k
        pred_emb_self_new = tf.reshape(pred_emb_self_new, [-1,len(tar_idx),k])

        return pred_emb_self_new, wgt, nlz_wgt

    # GME-G
    # src_idx_ngb - e.g., cre_id in ngb; ft_idx_ngb - other fts used to compute the similarity
    def gen_emb_from_self_and_ngb_gen(data_embed_concat, data_embed_concat_ngb):
        tf.stop_gradient(emb_mat)
        ######### self attr #########
        # none * (len(attr_idx)*k)
        attr_emb = get_sel_col(data_embed_concat, attr_idx)
        # none * (len(tar_idx)*k)
        pred_emb_self = gamma * tf.nn.tanh(tf.matmul(attr_emb, W_meta))

        # none * 1 * (len(tar_idx)*k)
        pred_emb_self_exp = tf.expand_dims(pred_emb_self, 1)
        # none * (max_n_ngb+1) * (len(tar_idx)*k)
        pred_emb_self_tile = tf.tile(pred_emb_self_exp, [1, max_n_ngb+1, 1])
        # convert to 2D [fold the first 2 dims]
        # (none*(max_n_ngb+1)) * (len(tar_idx)*k)
        pred_emb_self_2d = tf.reshape(pred_emb_self_tile, [-1, len(tar_idx)*k])

        ######### ngb #########
        # none * max_n_ngb * (len(attr_idx) * k)
        attr_emb_ngb_ori = get_sel_col_ngb(data_embed_concat_ngb, attr_idx)
        # none * 1 * (len(attr_idx)*k)
        attr_emb_exp = tf.expand_dims(attr_emb, 1)
        # none * (max_n_ngb + 1) * (len(attr_idx)*k)
        attr_emb_ngb = tf.concat([attr_emb_ngb_ori, attr_emb_exp], 1)

        # convert to 2D [fold the first 2 dims]
        attr_emb_ngb_2d = tf.reshape(attr_emb_ngb, [-1, len(attr_idx)*k])
        # (none*(max_n_ngb+1)) * (len(tar_idx)*k)
        pred_emb_ngb_2d = gamma * tf.nn.tanh(tf.matmul(attr_emb_ngb_2d, W_meta))

        ######### GAT #########
        # (none*(max_ngb+1)) * (len(tar_idx)*k)
        temp_self = tf.matmul(pred_emb_self_2d, W_gat)
        temp_ngb = tf.matmul(pred_emb_ngb_2d, W_gat)
        # (none*(max_ngb+1)) * 1
        wgt = tf.nn.leaky_relu(tf.matmul(tf.concat([temp_self, temp_ngb], 1), a_gat))
        wgt = tf.reshape(wgt, [-1, max_n_ngb+1, 1])
        nlz_wgt = tf.nn.softmax(wgt, dim=1)

        temp_ngb_re = tf.reshape(temp_ngb, [-1, max_n_ngb+1, len(tar_idx)*k])
        # none * (len(tar_idx)*k)
        pred_emb_self_new = tf.nn.elu(tf.reduce_sum(temp_ngb_re * nlz_wgt, 1))
        # none * len(tar_idx) * k
        pred_emb_self_new = tf.reshape(pred_emb_self_new, [-1,len(tar_idx),k])

        return pred_emb_self_new, wgt, nlz_wgt

    # GME-A
    # src_idx_ngb - e.g., cre_id in ngb; ft_idx_ngb - other fts used to compute the similarity
    def gen_emb_from_self_and_ngb_attr(data_embed_concat, data_embed_concat_ngb):
        tf.stop_gradient(emb_mat)
        ######### self attr #########
        # none * (len(attr_idx)*k)
        attr_emb = get_sel_col(data_embed_concat, attr_idx)
        # none * 1 * (len(attr_idx)*k)
        attr_emb_exp = tf.expand_dims(attr_emb, 1)
        # none * (max_n_ngb+1) * (len(attr_idx)*k)
        attr_emb_tile = tf.tile(attr_emb_exp, [1, max_n_ngb+1, 1])

        ######### ngb #########
        # none * max_n_ngb * (len(attr_idx) * k)
        attr_emb_ngb_ori = get_sel_col_ngb(data_embed_concat_ngb, attr_idx)
        # none * (max_n_ngb + 1) * (len(attr_idx)*k)
        attr_emb_ngb = tf.concat([attr_emb_ngb_ori, attr_emb_exp], 1)

        ######### GAT #########
        # convert to 2D [fold the first 2 dims]
        attr_emb_2d = tf.reshape(attr_emb_tile, [-1, len(attr_idx)*k])
        attr_emb_ngb_2d = tf.reshape(attr_emb_ngb, [-1, len(attr_idx)*k])

        # (none*(max_ngb+1)) * hidden_dim
        temp_self = tf.matmul(attr_emb_2d, W_gat)
        temp_ngb = tf.matmul(attr_emb_ngb_2d, W_gat)
        # (none*(max_ngb+1)) * 1
        wgt = tf.nn.leaky_relu(tf.matmul(tf.concat([temp_self, temp_ngb], 1), a_gat))
        wgt = tf.reshape(wgt, [-1, max_n_ngb+1, 1])

        nlz_wgt = tf.nn.softmax(wgt, dim=1)
        temp_ngb_re = tf.reshape(temp_ngb, [-1, max_n_ngb+1, att_dim])
        up_attr_emb = tf.nn.elu(tf.reduce_sum(temp_ngb_re * nlz_wgt, 1))

        pred_emb = gamma * tf.nn.tanh(tf.matmul(up_attr_emb, W_meta))
        pred_emb = tf.reshape(pred_emb, [-1,len(tar_idx),k])
        return pred_emb, wgt, nlz_wgt

    def get_concate_embed_w_meta(data_embed_concat, pred_emb):
        cur_slot_idx = 0
        if cur_slot_idx in tar_idx:
            cur_col_idx = tar_slot_map[cur_slot_idx]
            final_emb = pred_emb[:, cur_col_idx:cur_col_idx+1, :]
        else:
            final_emb = data_embed_concat[:, cur_slot_idx:cur_slot_idx+1, :]

        for i in range(1, n_slot):
            cur_slot_idx = i
            if cur_slot_idx in tar_idx:
                cur_col_idx = tar_slot_map[cur_slot_idx]
                cur_x = pred_emb[:, cur_col_idx:cur_col_idx+1, :]
            else:
                cur_x = data_embed_concat[:, cur_slot_idx:cur_slot_idx+1, :]
            final_emb = tf.concat([final_emb, cur_x], 1)
        return final_emb

    # input: (?, n_slot, k)
    # output: (?, 1)
    def get_y_hat(final_emb):
        # include output layer
        n_layer = len(layer_dim)
        data_embed_dnn = tf.reshape(final_emb, [-1, n_slot*k])
        cur_layer = data_embed_dnn
        # loop to create DNN struct
        for i in range(0, n_layer):
            # output layer, linear activation
            if i == n_layer - 1:
                cur_layer = tf.matmul(cur_layer, weight_dict[i]) #+ bias_dict[i]
            else:
                cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight_dict[i])) # + bias_dict[i])
                cur_layer = tf.nn.dropout(cur_layer, keep_prob)

        y_hat = cur_layer
        return y_hat

    def get_metric(test_pred_score_all, test_label_all, test_loss_all):
        test_pred_score_re = func.list_flatten(test_pred_score_all)
        test_label_re = func.list_flatten(test_label_all)
        test_auc, _, _ = func.cal_auc(test_pred_score_re, test_label_re)
        test_loss = np.mean(test_loss_all)
        return test_auc, test_loss

    ###########################################################
    x_input_a = tf.placeholder(tf.int32, shape=[None, total_num_ft_col])
    x_input_one_hot_a, x_input_mul_hot_a, x_input_one_hot_ngb_a, x_input_mul_hot_ngb_a \
        = partition_input(x_input_a)
    y_target_a = tf.placeholder(tf.float32, shape=[None, 1])

    x_input_b = tf.placeholder(tf.int32, shape=[None, total_num_ft_col])
    x_input_one_hot_b, x_input_mul_hot_b, _, _ \
        = partition_input(x_input_b)
    y_target_b = tf.placeholder(tf.float32, shape=[None, 1])

    # dropout keep prob
    keep_prob = tf.placeholder(tf.float32)

    ############################
    # emb_mat dim add 1 -> for padding (idx = 0)
    with tf.device('/cpu:0'):
        emb_mat = tf.Variable(tf.random_normal([n_ft + 1, k], stddev=0.01))

    if meta_mode == 'self':
        in_dim = len(attr_idx)*k
        out_dim = len(tar_idx)*k
        cur_range = np.sqrt(6.0 / (in_dim + out_dim))
        W_meta = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))
        meta_vars = [W_meta]
    elif meta_mode == 'ngb':
        in_dim = len(tar_idx)*k
        out_dim = len(tar_idx)*k
        cur_range = np.sqrt(6.0 / (in_dim + out_dim))
        W_meta = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))
        meta_vars = [W_meta]
    elif meta_mode == 'GME-A':
        in_dim = att_dim
        out_dim = len(tar_idx)*k
        cur_range = np.sqrt(6.0 / (in_dim + out_dim))
        W_meta = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))

        in_dim = len(attr_idx)*k
        out_dim = att_dim
        cur_range = np.sqrt(6.0 / (in_dim + out_dim))
        W_gat = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))

        in_dim = 2*att_dim
        out_dim = 1
        cur_range = np.sqrt(6.0 / (in_dim + out_dim))
        a_gat = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))
        # var list of GME
        meta_vars = [W_meta, W_gat, a_gat]
    elif meta_mode == 'GME-P' or meta_mode == 'GME-G':
        in_dim = len(attr_idx)*k
        out_dim = len(tar_idx)*k
        cur_range = np.sqrt(6.0 / (in_dim + out_dim))
        W_meta = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))

        in_dim = len(tar_idx)*k
        out_dim = len(tar_idx)*k
        cur_range = np.sqrt(6.0 / (in_dim + out_dim))
        W_gat = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))

        in_dim = 2*len(tar_idx)*k
        out_dim = 1
        cur_range = np.sqrt(6.0 / (in_dim + out_dim))
        a_gat = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))

        # var list of GME
        meta_vars = [W_meta, W_gat, a_gat]


    ################################
    # include output layer
    n_layer = len(layer_dim)
    in_dim = n_slot*k
    weight_dict={}

    # loop to create DNN vars
    for i in range(0, n_layer):
        out_dim = layer_dim[i]
        cur_range = np.sqrt(6.0 / (in_dim + out_dim))
        weight_dict[i] = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))
#         bias_dict[i] = tf.Variable(tf.constant(0.0, shape=[out_dim]))
        in_dim = layer_dim[i]

    ##############################
    # Step 1: cold-start
    #     use the generated embeddings to make predictions
    #     and calculate the cold-start loss_a
    ####### DNN ########
    data_embed_concat_a = get_concate_embed(x_input_one_hot_a, x_input_mul_hot_a)
    y_hat = get_y_hat(data_embed_concat_a)
    warm_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=y_target_a))

    if meta_mode == 'self':
        pred_emb_a = gen_emb_from_self(data_embed_concat_a)
    elif meta_mode == 'ngb':
        data_embed_concat_ngb_a = get_concate_embed_ngb(x_input_one_hot_ngb_a, x_input_mul_hot_ngb_a)
        pred_emb_a = get_emb_from_ngb(data_embed_concat_ngb_a, x_input_one_hot_ngb_a)
    elif meta_mode == 'GME-P':
        data_embed_concat_ngb_a = get_concate_embed_ngb(x_input_one_hot_ngb_a, x_input_mul_hot_ngb_a)
        pred_emb_a, wgt_a, nlz_wgt_a = gen_emb_from_self_and_ngb_pre(data_embed_concat_a, data_embed_concat_ngb_a)
    elif meta_mode == 'GME-G':
        data_embed_concat_ngb_a = get_concate_embed_ngb(x_input_one_hot_ngb_a, x_input_mul_hot_ngb_a)
        pred_emb_a, wgt_a, nlz_wgt_a = gen_emb_from_self_and_ngb_gen(data_embed_concat_a, data_embed_concat_ngb_a)
    elif meta_mode == 'GME-A':
        data_embed_concat_ngb_a = get_concate_embed_ngb(x_input_one_hot_ngb_a, x_input_mul_hot_ngb_a)
        pred_emb_a, wgt_a, nlz_wgt_a = gen_emb_from_self_and_ngb_attr(data_embed_concat_a, data_embed_concat_ngb_a)

    final_emb_a = get_concate_embed_w_meta(data_embed_concat_a, pred_emb_a)
    cold_y_hat_a = get_y_hat(final_emb_a)
    cold_loss_a = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cold_y_hat_a, labels=y_target_a))

    ###############
    # Step 2: apply gradient descent once
    #     get the adapted embedding
    cold_emb_grads = tf.gradients(cold_loss_a, pred_emb_a)[0]
    pred_emb_a_new = pred_emb_a - cold_eta * cold_emb_grads

    ###############
    # Step 3:
    #     use the adapted embedding to make prediction on another mini-batch
    #     and calculate the warm-up loss_b
    data_embed_concat_b = get_concate_embed(x_input_one_hot_b, x_input_mul_hot_b)
    final_emb_b = get_concate_embed_w_meta(data_embed_concat_b, pred_emb_a_new)
    cold_y_hat_b = get_y_hat(final_emb_b)
    cold_loss_b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cold_y_hat_b, labels=y_target_b))

    ###############
    # Step 4: calculate the final loss
    meta_loss = cold_loss_a * alpha + cold_loss_b * (1-alpha)

    #############################
    # prediction
    #############################
    pred_score = tf.sigmoid(y_hat)
    pred_score_a = tf.sigmoid(cold_y_hat_a)

    if opt_alg == 'Adam':
        meta_optimizer = tf.train.AdamOptimizer(meta_eta).minimize(meta_loss, var_list=meta_vars)
    else:
        meta_optimizer = tf.train.AdagradOptimizer(meta_eta).minimize(meta_loss, var_list=meta_vars)

    ########################################
    # Launch the graph.
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        train_loss_list = []

        func.print_time()
        print('Start train loop')

        ######################################
        # A: load saved model
        # A1: test ori model
        save_dict = {}
        save_dict['emb_mat'] = emb_mat
        for i in range(0, n_layer):
            cur_key = 'weight_dict[' + str(i) + ']'
            save_dict[cur_key] = weight_dict[i]
        saver = tf.train.Saver(save_dict)

        saver.restore(sess, model_loading_addr)

        ######################################
        # B: train meta - only update W_meta
        # B1: gen meta - assign [a new tensor] - test new meta
        epoch = -1
        try:
            while True:
                epoch += 1
                train_ft_inst_a, train_label_inst_a = sess.run([train_ft_a, train_label_a])
                train_ft_inst_b, train_label_inst_b = sess.run([train_ft_b, train_label_b])
                sess.run(meta_optimizer, feed_dict={x_input_a:train_ft_inst_a, y_target_a:train_label_inst_a, \
                                                  x_input_b:train_ft_inst_b, y_target_b:train_label_inst_b, \
                                                  keep_prob:kp_prob})

                # record loss and accuracy every step_size generations
                if (epoch+1)%record_step_size == 0:
                    train_loss_temp = sess.run(meta_loss, feed_dict={ \
                                            x_input_a:train_ft_inst_a, y_target_a:train_label_inst_a, \
                                            x_input_b:train_ft_inst_b, y_target_b:train_label_inst_b, \
                                            keep_prob:1.0})
                    train_loss_list.append(train_loss_temp)

                    auc_and_loss = [epoch+1, train_loss_temp]
                    auc_and_loss = [np.round(xx,4) for xx in auc_and_loss]
                    func.print_time()
                    print('Generation # {}. Train Loss: {:.4f}.'\
                          .format(*auc_and_loss))

        except tf.errors.OutOfRangeError:
            func.print_time()
            print('Done training meta -- epoch limit reached')

        test_pred_score_all = []
        test_label_all = []
        test_loss_all = []
        test_pred_score_all_meta = []
        test_loss_all_meta = []

        try:
            while True:
                test_ft_inst, test_label_inst = sess.run([test_ft, test_label])
                cur_test_pred_score, cur_test_pred_score_meta = \
                                sess.run([pred_score, pred_score_a], feed_dict={ \
                                x_input_a:test_ft_inst, keep_prob:1.0})

                cur_test_loss, cur_test_loss_meta = sess.run([warm_loss, cold_loss_a], feed_dict={ \
                                        x_input_a:test_ft_inst, \
                                        y_target_a:test_label_inst, keep_prob:1.0})

                test_pred_score_all.append(cur_test_pred_score.flatten())
                test_pred_score_all_meta.append(cur_test_pred_score_meta.flatten())
                test_label_all.append(test_label_inst)
                test_loss_all.append(cur_test_loss)
                test_loss_all_meta.append(cur_test_loss_meta)

        except tf.errors.OutOfRangeError:
            func.print_time()
            print('Done testing meta -- epoch limit reached')

        finally:
            coord.request_stop()
        coord.join(threads)

        # calculate metric
        test_auc, test_loss = get_metric(test_pred_score_all, test_label_all, test_loss_all)
        test_auc_meta, test_loss_meta = get_metric(test_pred_score_all_meta, test_label_all, test_loss_all_meta)

        # append to result_list
        result_list.append([meta_batch_size, cold_eta, meta_eta, test_auc, test_loss, \
                            test_auc_meta, test_loss_meta])

print('*'*20)
print('meta_mode: ' + meta_mode)

header_row = ['meta_bs', 'cold_eta', 'meta_eta', 'auc', 'loss', 'auc_meta', 'loss_meta']

fmt_str = '{:<10}'*len(header_row)
print(fmt_str.format(*header_row))

fmt_str = '{:<10.6f}'*len(header_row)
for i in range(len(result_list)):
    tmp = result_list[i]
    print(fmt_str.format(*tmp))

