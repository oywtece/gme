import numpy as np
import tensorflow as tf
import datetime
import ctr_funcs as func
import config_gme as cfg
from time import time
import os
import shutil

# whether to perform warm up
warm_up_bool = cfg.warm_up_bool

test_batch_size = cfg.test_batch_size
meta_mode = cfg.meta_mode
alpha = cfg.alpha
gamma = cfg.gamma

train_file_name_warm = cfg.train_file_name_warm
train_file_name_warm_2 = cfg.train_file_name_warm_2
n_epoch = cfg.n_epoch

label_col_idx = 0
num_csv_col_warm = cfg.num_csv_col_warm
total_num_ft_col_warm = num_csv_col_warm - 1

num_csv_col_w_ngb = cfg.num_csv_col_w_ngb
total_num_ft_col_cold = num_csv_col_w_ngb - 1

# config
# must be from small to large
tar_idx = cfg.tar_idx
attr_idx = cfg.attr_idx

str_txt = cfg.output_file_name
base_path = './tmp'
model_loading_addr = cfg.model_loading_addr
model_saving_addr = base_path + '/meta_' + str_txt + '/'
output_file_name = base_path + '/meta_' + str_txt + '.txt'
save_model_ind = cfg.save_model_ind
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

eta_range = cfg.eta_range
meta_batch_size_range = cfg.meta_batch_size_range
cold_eta_range = cfg.cold_eta_range
meta_eta_range = cfg.meta_eta_range

# key: slot_idx in ori data, val: col_idx in pred_emb
tar_slot_map = {}
for i in range(len(tar_idx)):
    tar_slot_map[tar_idx[i]] = i

## create para list
para_list = []
for i in range(len(eta_range)):
    for ii in range(len(meta_batch_size_range)):
        for iii in range(len(cold_eta_range)):
            for iv in range(len(meta_eta_range)):
                para_list.append([eta_range[i], meta_batch_size_range[ii], cold_eta_range[iii], \
                                  meta_eta_range[iv]])

## record results
result_list = []

# loop over para_list
for item in para_list:
    eta = item[0]
    meta_batch_size = item[1]
    cold_eta = item[2]
    meta_eta = item[3]

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
    tf.set_random_seed(123)

    if input_format == 'tfrecord':
        if warm_up_bool:
            train_ft_warm, train_label_warm = func.tfrecord_input_pipeline_test(train_file_name_warm, num_csv_col_warm, batch_size, n_epoch)
            train_ft_meta_warm, train_label_meta_warm = func.tfrecord_input_pipeline_test(train_file_name_warm, num_csv_col_warm, batch_size, n_epoch)

            test_ft_warm, test_label_warm = func.tfrecord_input_pipeline_test(test_file_name, num_csv_col_w_ngb, test_batch_size, 1)
            test_ft_meta_warm, test_label_meta_warm = func.tfrecord_input_pipeline_test(test_file_name, num_csv_col_w_ngb, test_batch_size, 1)

            test_ft_copy, test_label_copy = func.tfrecord_input_pipeline_test(test_file_name, num_csv_col_w_ngb, test_batch_size, 1)
            # warm up 2
            train_ft_warm_2, train_label_warm_2 = func.tfrecord_input_pipeline_test(train_file_name_warm_2, num_csv_col_warm, batch_size, n_epoch)
            train_ft_meta_warm_2, train_label_meta_warm_2 = func.tfrecord_input_pipeline_test(train_file_name_warm_2, num_csv_col_warm, batch_size, n_epoch)

            test_ft_warm_2, test_label_warm_2 = func.tfrecord_input_pipeline_test(test_file_name, num_csv_col_w_ngb, test_batch_size, 1)
            test_ft_meta_warm_2, test_label_meta_warm_2 = func.tfrecord_input_pipeline_test(test_file_name, num_csv_col_w_ngb, test_batch_size, 1)

        train_ft_a, train_label_a = func.tfrecord_input_pipeline_test(train_file_name_a, num_csv_col_w_ngb, meta_batch_size, n_epoch_meta)
        train_ft_b, train_label_b = func.tfrecord_input_pipeline_test(train_file_name_b, num_csv_col_w_ngb, meta_batch_size, n_epoch_meta)

        test_ft, test_label = func.tfrecord_input_pipeline_test(test_file_name, num_csv_col_w_ngb, test_batch_size, 1)
        test_ft_meta, test_label_meta = func.tfrecord_input_pipeline_test(test_file_name, num_csv_col_w_ngb, test_batch_size, 1)

    ########################################################################
    ########################################################################
    def partition_input(x_input):
        idx_1 = n_one_hot_slot
        idx_2 = idx_1 + n_mul_hot_slot*max_len_per_slot
        x_input_one_hot = x_input[:, 0:idx_1]
        x_input_mul_hot = x_input[:, idx_1:idx_2]
        # shape=[None, n_mul_hot_slot, max_len_per_slot]
        x_input_mul_hot = tf.reshape(x_input_mul_hot, (-1, n_mul_hot_slot, max_len_per_slot))
        return x_input_one_hot, x_input_mul_hot

    # data format (label is removed from x_input)
    # tar, ngb (w diff n_fts)
    def partition_input_w_ngb(x_input):
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
        ######### self attr #########
        # none * (len(attr_idx)*k)
        attr_emb = get_sel_col(data_embed_concat, attr_idx)
        # none * (len(tar_idx)*k)
        pred_emb_self = gamma * tf.nn.tanh(tf.matmul(attr_emb, W_meta))
        pred_emb_self = tf.reshape(pred_emb_self, [-1,len(tar_idx),k])
        return pred_emb_self

    def get_emb_from_ngb(data_embed_concat_ngb, x_input_one_hot_ngb):
        # none * max_n_ngb * (len(tar_idx) * k)
        ngb_emb = get_sel_col_ngb(data_embed_concat_ngb, tar_idx)
        n_valid_ngb = count_n_valid_ngb(x_input_one_hot_ngb)
        # must flatten first, otherwise [*,a,b] / [*,c] will result in err
        avg_ngb_emb = tf.layers.flatten(tf.reduce_sum(ngb_emb, 1)) / (n_valid_ngb + 1e-5)
        pred_emb = gamma * tf.nn.tanh(tf.matmul(avg_ngb_emb, W_meta))
        pred_emb = tf.reshape(pred_emb, [-1,len(tar_idx),k])
        return pred_emb

    # based on GAT
    # src_idx_ngb - e.g., cre_id in ngb; ft_idx_ngb - other fts used to compute the similarity
    def gen_emb_from_self_and_ngb_pre(data_embed_concat, data_embed_concat_ngb):
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

    # based on GAT
    # src_idx_ngb - e.g., cre_id in ngb; ft_idx_ngb - other fts used to compute the similarity
    def gen_emb_from_self_and_ngb_gen(data_embed_concat, data_embed_concat_ngb):
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

    # based on GAT
    # src_idx_ngb - e.g., cre_id in ngb; ft_idx_ngb - other fts used to compute the similarity
    def gen_emb_from_self_and_ngb_attr(data_embed_concat, data_embed_concat_ngb):
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
    if warm_up_bool:
        x_input_warm = tf.placeholder(tf.int32, shape=[None, total_num_ft_col_warm])
        x_input_one_hot_warm, x_input_mul_hot_warm = partition_input(x_input_warm)
        y_target_warm = tf.placeholder(tf.float32, shape=[None, 1])

    x_input_a = tf.placeholder(tf.int32, shape=[None, total_num_ft_col_cold])
    x_input_one_hot_a, x_input_mul_hot_a, x_input_one_hot_ngb_a, x_input_mul_hot_ngb_a \
        = partition_input_w_ngb(x_input_a)
    y_target_a = tf.placeholder(tf.float32, shape=[None, 1])

    x_input_b = tf.placeholder(tf.int32, shape=[None, total_num_ft_col_cold])
    x_input_one_hot_b, x_input_mul_hot_b, _, _ \
        = partition_input_w_ngb(x_input_b)
    y_target_b = tf.placeholder(tf.float32, shape=[None, 1])
    
    # dropout keep prob
    keep_prob = tf.placeholder(tf.float32)

    ############################
    # emb_mat dim add 1 -> for padding (idx = 0)
    with tf.device('/cpu:0'):
        emb_mat = tf.Variable(tf.random_normal([n_ft + 1, k], stddev=0.01))

    if warm_up_bool:
        # placeholder for new emb_mat
        emb_mat_input = tf.placeholder(tf.float32, shape=[n_ft + 1, k])
        emb_mat_assign_op = tf.assign(emb_mat, emb_mat_input)

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
        # var list of meta emb
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

        # var list of meta emb
        meta_vars = [W_meta, W_gat, a_gat]

    ################################
    # include output layer
    n_layer = len(layer_dim)
    in_dim = n_slot*k
    weight_dict={}
#     bias_dict={}

    # loop to create DNN vars
    for i in range(0, n_layer):
        out_dim = layer_dim[i]
        cur_range = np.sqrt(6.0 / (in_dim + out_dim))
        weight_dict[i] = tf.Variable(tf.random_uniform([in_dim, out_dim], -cur_range, cur_range))
        in_dim = layer_dim[i]

    ####### DNN ########
    if warm_up_bool:
        data_embed_concat_warm = get_concate_embed(x_input_one_hot_warm, x_input_mul_hot_warm)
        y_hat_warm = get_y_hat(data_embed_concat_warm)
        # used for training
        warm_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat_warm, labels=y_target_warm))
        warm_vars = [emb_mat]

    ##############################
    # Step 1: cold-start
    #     use the generated embeddings to make predictions
    #     and calculate the cold-start loss_a

    data_embed_concat_a = get_concate_embed(x_input_one_hot_a, x_input_mul_hot_a)
    y_hat = get_y_hat(data_embed_concat_a)
    # used for eval only
    eval_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=y_target_a))

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
    pred_score = tf.sigmoid(y_hat) # using ori emb
    pred_score_a = tf.sigmoid(cold_y_hat_a) # using new, gen emb

    if opt_alg == 'Adam':
        meta_optimizer = tf.train.AdamOptimizer(meta_eta).minimize(meta_loss, var_list=meta_vars)
        if warm_up_bool:
            warm_optimizer = tf.train.AdamOptimizer(eta).minimize(warm_loss, var_list=warm_vars)
    else:
        meta_optimizer = tf.train.AdagradOptimizer(meta_eta).minimize(meta_loss, var_list=meta_vars)
        if warm_up_bool:
            warm_optimizer = tf.train.AdagradOptimizer(eta).minimize(warm_loss, var_list=warm_vars)

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
        test_pred_score_all = []
        test_label_all = []
        test_loss_all = []

        test_pred_score_all_meta = []
        test_label_all_meta = []
        test_loss_all_meta = []

        if warm_up_bool:
            test_pred_score_all_warm = []
            test_label_all_warm = []
            test_loss_all_warm = []

            test_pred_score_all_meta_warm = []
            test_label_all_meta_warm = []
            test_loss_all_meta_warm = []

            test_pred_score_all_warm_2 = []
            test_label_all_warm_2 = []
            test_loss_all_warm_2 = []

            test_pred_score_all_meta_warm_2 = []
            test_label_all_meta_warm_2 = []
            test_loss_all_meta_warm_2 = []

        save_dict = {}
        save_dict['emb_mat'] = emb_mat
        for i in range(0, n_layer):
            cur_key = 'weight_dict[' + str(i) + ']'
            save_dict[cur_key] = weight_dict[i]
        saver = tf.train.Saver(save_dict)

        saver.restore(sess, model_loading_addr)

        ######################################
        # A: test directly
        ######################################
        try:
            while True:
                test_ft_inst, test_label_inst = sess.run([test_ft, test_label])
                cur_test_pred_score = sess.run(pred_score, feed_dict={ \
                                        x_input_a:test_ft_inst, keep_prob:1.0})
                test_pred_score_all.append(cur_test_pred_score.flatten())
                test_label_all.append(test_label_inst)

                cur_test_loss = sess.run(eval_loss, feed_dict={ \
                                        x_input_a:test_ft_inst, \
                                        y_target_a:test_label_inst, keep_prob:1.0})
                test_loss_all.append(cur_test_loss)
        except tf.errors.OutOfRangeError:
            func.print_time()
            print('Done direct testing')

        if warm_up_bool:
            ######################################
            # B1: warm up training - update emb_mat
            ######################################
            try:
                while True:
                    train_ft_inst_warm, train_label_inst_warm = sess.run([train_ft_warm, train_label_warm])
                    # run warm optimizer
                    sess.run(warm_optimizer, feed_dict={x_input_warm:train_ft_inst_warm, y_target_warm:train_label_inst_warm, \
                                                        keep_prob:kp_prob})
            except tf.errors.OutOfRangeError:
                func.print_time()
                print('Done warm up training')

            ######################################
            # B2: warm up testing
            ######################################
            try:
                while True:
                    test_ft_inst, test_label_inst = sess.run([test_ft_warm, test_label_warm])
                    cur_test_pred_score = sess.run(pred_score, feed_dict={ \
                                            x_input_a:test_ft_inst, keep_prob:1.0})
                    test_pred_score_all_warm.append(cur_test_pred_score.flatten())
                    test_label_all_warm.append(test_label_inst)

                    cur_test_loss = sess.run(eval_loss, feed_dict={x_input_a:test_ft_inst, \
                                            y_target_a:test_label_inst, keep_prob:1.0})
                    test_loss_all_warm.append(cur_test_loss)

            except tf.errors.OutOfRangeError:
                func.print_time()
                print('Done warm up testing')


            ######################################
            # B3: warm up training - update emb_mat
            ######################################
            try:
                while True:
                    train_ft_inst_warm, train_label_inst_warm = sess.run([train_ft_warm_2, train_label_warm_2])
                    # run warm optimizer
                    sess.run(warm_optimizer, feed_dict={x_input_warm:train_ft_inst_warm, y_target_warm:train_label_inst_warm, \
                                                        keep_prob:kp_prob})
            except tf.errors.OutOfRangeError:
                func.print_time()
                print('Done warm up training 2')

            ######################################
            # B4: warm up testing
            ######################################
            try:
                while True:
                    test_ft_inst, test_label_inst = sess.run([test_ft_warm_2, test_label_warm_2])
                    cur_test_pred_score = sess.run(pred_score, feed_dict={ \
                                            x_input_a:test_ft_inst, keep_prob:1.0})
                    test_pred_score_all_warm_2.append(cur_test_pred_score.flatten())
                    test_label_all_warm_2.append(test_label_inst)

                    cur_test_loss = sess.run(eval_loss, feed_dict={x_input_a:test_ft_inst, \
                                            y_target_a:test_label_inst, keep_prob:1.0})
                    test_loss_all_warm_2.append(cur_test_loss)

            except tf.errors.OutOfRangeError:
                func.print_time()
                print('Done warm up testing 2')

        ######################################
        # C1: meta training - update GME params
        ######################################
        # reload model params before warm up
        saver.restore(sess, model_loading_addr)

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
            print('Done meta training')

        ######################################
        # C2: meta testing
        ######################################
        try:
            while True:
                test_ft_inst, test_label_inst = sess.run([test_ft_meta, test_label_meta])
                cur_test_pred_score = sess.run(pred_score_a, feed_dict={ \
                                x_input_a:test_ft_inst, keep_prob:1.0})
                test_pred_score_all_meta.append(cur_test_pred_score.flatten())
                test_label_all_meta.append(test_label_inst)

                cur_test_loss = sess.run(cold_loss_a, feed_dict={ \
                                        x_input_a:test_ft_inst, \
                                        y_target_a:test_label_inst, keep_prob:1.0})
                test_loss_all_meta.append(cur_test_loss)

        except tf.errors.OutOfRangeError:
            func.print_time()
            print('Done meta testing')

        if warm_up_bool:
            ######################################
            # D0: update emb_mat of new ads by GME
            ######################################
            try:
                # get emb_mat
                emb_mat_val = sess.run(emb_mat)
                while True:
                    test_ft_inst_copy, test_label_inst_copy = sess.run([test_ft_copy, test_label_copy])
                    # get new ID embs
                    pred_emb_w_val = sess.run(pred_emb_a, feed_dict={x_input_a:test_ft_inst_copy})
                    # assume only 1 tar idx
                    id_col = test_ft_inst_copy[:, tar_idx]
                    for iter_ee in range(len(id_col)):
                        cur_ft_id = id_col[iter_ee]
                        cur_emb = pred_emb_w_val[iter_ee,:]
                        emb_mat_val[cur_ft_id,:] = cur_emb
            except tf.errors.OutOfRangeError:
                # update emb_mat
                sess.run(emb_mat_assign_op, feed_dict={emb_mat_input:emb_mat_val})
                func.print_time()
                print('Done update emb_mat with meta')

            ######################################
            # D1: warm up training after meta
            ######################################
            try:
                while True:
                    train_ft_inst_warm, train_label_inst_warm = sess.run([train_ft_meta_warm, train_label_meta_warm])
                    # run warm optimizer
                    sess.run(warm_optimizer, feed_dict={x_input_warm:train_ft_inst_warm, y_target_warm:train_label_inst_warm, \
                                                      keep_prob:kp_prob})
            except tf.errors.OutOfRangeError:
                func.print_time()
                print('Done warm up training after meta')

            ######################################
            # D2: warm up testing after meta
            ######################################
            try:
                while True:
                    test_ft_inst, test_label_inst = sess.run([test_ft_meta_warm, test_label_meta_warm])
                    cur_test_pred_score = sess.run(pred_score, feed_dict={ \
                                            x_input_a:test_ft_inst, keep_prob:1.0})
                    test_pred_score_all_meta_warm.append(cur_test_pred_score.flatten())
                    test_label_all_meta_warm.append(test_label_inst)

                    cur_test_loss = sess.run(eval_loss, feed_dict={ \
                                            x_input_a:test_ft_inst, \
                                            y_target_a:test_label_inst, keep_prob:1.0})
                    test_loss_all_meta_warm.append(cur_test_loss)

            except tf.errors.OutOfRangeError:
                func.print_time()
                print('Done warm up testing after meta')

            ######################################
            # D3: warm up training after meta
            ######################################
            try:
                while True:
                    train_ft_inst_warm, train_label_inst_warm = sess.run([train_ft_meta_warm_2, train_label_meta_warm_2])
                    # run warm optimizer
                    sess.run(warm_optimizer, feed_dict={x_input_warm:train_ft_inst_warm, y_target_warm:train_label_inst_warm, \
                                                      keep_prob:kp_prob})
            except tf.errors.OutOfRangeError:
                func.print_time()
                print('Done warm up training after meta 2')

            ######################################
            # D4: warm up testing after meta
            ######################################
            try:
                while True:
                    test_ft_inst, test_label_inst = sess.run([test_ft_meta_warm_2, test_label_meta_warm_2])
                    cur_test_pred_score = sess.run(pred_score, feed_dict={ \
                                            x_input_a:test_ft_inst, keep_prob:1.0})
                    test_pred_score_all_meta_warm_2.append(cur_test_pred_score.flatten())
                    test_label_all_meta_warm_2.append(test_label_inst)

                    cur_test_loss = sess.run(eval_loss, feed_dict={ \
                                            x_input_a:test_ft_inst, \
                                            y_target_a:test_label_inst, keep_prob:1.0})
                    test_loss_all_meta_warm_2.append(cur_test_loss)

            except tf.errors.OutOfRangeError:
                func.print_time()
                print('Done warm up testing after meta 2')

        #############################
        # dummy opt to pass syntax check
        # otherwise, you have
        # try
        # except
        # if [-> syntax error]
        # finally
        try:
            (aa,bb) = (2,1)
            cc = aa/bb
        except ZeroDivisionError:
            print('divide by zero')
        #############################

        finally:
            coord.request_stop()
        coord.join(threads)

        # calculate metric
        test_auc, test_loss = get_metric(test_pred_score_all, test_label_all, test_loss_all)
        test_auc_meta, test_loss_meta = get_metric(test_pred_score_all_meta, test_label_all_meta, test_loss_all_meta)

        if warm_up_bool:
            test_auc_warm, test_loss_warm = get_metric(test_pred_score_all_warm, test_label_all_warm, test_loss_all_warm)
            test_auc_meta_warm, test_loss_meta_warm = get_metric(test_pred_score_all_meta_warm, test_label_all_meta_warm, test_loss_all_meta_warm)
            test_auc_warm_2, test_loss_warm_2 = get_metric(test_pred_score_all_warm_2, test_label_all_warm_2, test_loss_all_warm_2)
            test_auc_meta_warm_2, test_loss_meta_warm_2 = get_metric(test_pred_score_all_meta_warm_2, test_label_all_meta_warm_2, test_loss_all_meta_warm_2)

            result_list.append([batch_size, meta_batch_size, eta, cold_eta, meta_eta, \
                                test_auc, test_loss, \
                                test_auc_meta, test_loss_meta, \
                                test_auc_warm, test_loss_warm, \
                                test_auc_meta_warm, test_loss_meta_warm, \
                                test_auc_warm_2, test_loss_warm_2, \
                                test_auc_meta_warm_2, test_loss_meta_warm_2])
        else:
            result_list.append([meta_batch_size, cold_eta, meta_eta, \
                                test_auc, test_loss, \
                                test_auc_meta, test_loss_meta])

if warm_up_bool:
    header_row = ['bs', 'meta_bs', 'eta', 'cold_eta', 'meta_eta', \
                  'auc', 'loss', \
                  'auc_m', 'loss_m', \
                  'auc_w', 'loss_w', \
                  'auc_mw', 'loss_mw', \
                  'auc_w2', 'loss_w2', \
                  'auc_mw2', 'loss_mw2']
else:
    header_row = ['meta_bs', 'cold_eta', 'meta_eta', \
                  'auc', 'loss', \
                  'auc_m', 'loss_m']

print('*'*20)
print('meta_mode: ' + meta_mode)

fmt_str = '{:<10}'*len(header_row)
print(fmt_str.format(*header_row))

fmt_str = '{:<10.5f}'*len(header_row)
for i in range(len(result_list)):
    tmp = result_list[i]
    print(fmt_str.format(*tmp))

