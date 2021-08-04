import tensorflow as tf
import numpy as np
import datetime
from sklearn import metrics

def cal_auc(pred_score, label):
    fpr, tpr, thresholds = metrics.roc_curve(label, pred_score, pos_label=1)
    auc_val = metrics.auc(fpr, tpr)
    return auc_val, fpr, tpr

def cal_group_auc(pred_score, label, uid):
    uid_dict = {}
    for i in range(len(uid)):
        cur_uid = uid[i]
        cur_score = pred_score[i]
        cur_label = label[i]
        
        if cur_uid not in uid_dict:
            u_pred_score_list = []
            u_label_list = []
            uid_dict[cur_uid] = [u_pred_score_list, u_label_list]
        val = uid_dict[cur_uid]
        val[0].append(cur_score)
        val[1].append(cur_label)
        uid_dict[cur_uid] = val
    
    auc_val_list = []
    for key in uid_dict:
        val = uid_dict[key]
        u_pred_score_list = val[0]
        u_label_list = val[1]
        fpr, tpr, _ = metrics.roc_curve(u_label_list, u_pred_score_list, pos_label=1)
        auc_val = metrics.auc(fpr, tpr)        
#        if np.isnan(auc_val):
#           auc_val = 1.0
        auc_val_list.append(auc_val)
    
    group_auc = np.nanmean(auc_val_list)
    return group_auc

# 1 - pred/true
def cal_bias(pred_score, label):
    ctr = np.sum(label) / len(label)
    pctr = np.mean(pred_score)
    bias = 1.0 - pctr/(ctr+1e-6)
    return bias, ctr, pctr

def cal_rmse(pred_score, label):
    mse = metrics.mean_squared_error(label, pred_score)
    rmse = np.sqrt(mse)
    return rmse

def cal_rectified_rmse(pred_score, label, sample_rate):
    for idx, item in enumerate(pred_score):
        pred_score[idx] = item/(item + (1-item)/sample_rate)
    mse = metrics.mean_squared_error(label, pred_score)
    rmse = np.sqrt(mse)
    return rmse

# only works for 2D list
def list_flatten(input_list):
    output_list = [yy for xx in input_list for yy in xx]
    return output_list


def count_lines(file_name):
    num_lines = sum(1 for line in open(file_name, 'rt'))
    return num_lines

# this func is only for avito data
def tf_read_data(file_name_queue, label_col_idx, record_defaults):
    reader = tf.TextLineReader()
    key, value = reader.read(file_name_queue)
    
    # Default values, in case of empty columns. Also specifies the type of the decoded result.
    cols = tf.decode_csv(value, record_defaults=record_defaults)
    # you can only process the data using tf ops
    label = cols.pop(label_col_idx)
    feature = cols
    # Retrieve a single instance
    return feature, label

def tf_read_data_wo_label(file_name_queue, record_defaults):
    reader = tf.TextLineReader()
    key, value = reader.read(file_name_queue)
    # Default values, in case of empty columns. Also specifies the type of the decoded result.
    cols = tf.decode_csv(value, record_defaults=record_defaults)
    # you can only process the data using tf ops
    feature = cols
    # Retrieve a single instance
    return feature

# load training data
record_defaults = [[0]]*141
record_defaults[0] = [0.0]
def tf_input_pipeline(file_names, batch_size, num_epochs=1, label_col_idx=0, record_defaults=record_defaults):
    # if shffule = True -> shuffle over files; otherwise, train files one by one
    file_name_queue = tf.train.string_input_producer(file_names, num_epochs=num_epochs, shuffle=False)
    feature, label = tf_read_data(file_name_queue, label_col_idx, record_defaults)
    # min_after_dequeue defines how big a buffer we will randomly sample from
    # capacity must be larger than min_after_dequeue and the amount larger determines the max we
    # will prefetch
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3*batch_size
    feature_batch, label_batch = tf.train.shuffle_batch([feature, label], \
            batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    return feature_batch, label_batch

# without label
def tf_input_pipeline_wo_label(file_names, batch_size, num_epochs=1, record_defaults=record_defaults):
    # shuffle over files
    file_name_queue = tf.train.string_input_producer(file_names, num_epochs=num_epochs, shuffle=False)
    feature = tf_read_data_wo_label(file_name_queue, record_defaults)
    # min_after_dequeue defines how big a buffer we will randomly sample from
    # capacity must be larger than min_after_dequeue and the amount larger determines the max we
    # will prefetch
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3*batch_size
    feature_batch = tf.train.shuffle_batch([feature], \
            batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    return feature_batch

def tf_input_pipeline_test(file_names, batch_size, num_epochs=1, label_col_idx=0, record_defaults=record_defaults):
    # shuffle over files
    file_name_queue = tf.train.string_input_producer(file_names, num_epochs=num_epochs, shuffle=False)
    feature, label = tf_read_data(file_name_queue, label_col_idx, record_defaults)
    # min_after_dequeue defines how big a buffer we will randomly sample from
    # capacity must be larger than min_after_dequeue and the amount larger determines the max we
    # will prefetch
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3*batch_size
    feature_batch, label_batch = tf.train.batch([feature, label], \
            batch_size=batch_size, capacity=capacity)
    return feature_batch, label_batch

# num_csv_col -> in csv file
def tfrecord_input_pipeline(file_names, num_csv_col, batch_size, num_epochs=1):
    file_name_queue = tf.train.string_input_producer(file_names, num_epochs=num_epochs, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_name_queue)
    feature_set = {'label': tf.FixedLenFeature([1], tf.int64), \
                    'data': tf.FixedLenFeature([num_csv_col-1], tf.int64)}
    parsed_feature = tf.parse_single_example(serialized_example, features=feature_set)
    parsed_label = parsed_feature['label']
    parsed_data = parsed_feature['data']
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3*batch_size
    feature_batch, label_batch = tf.train.shuffle_batch([parsed_data, parsed_label], \
            batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    return feature_batch, label_batch    
    
def tfrecord_input_pipeline_test(file_names, num_csv_col, batch_size, num_epochs=1):
    file_name_queue = tf.train.string_input_producer(file_names, num_epochs=num_epochs, shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_name_queue)
    feature_set = {'label': tf.FixedLenFeature([1], tf.int64), \
                    'data': tf.FixedLenFeature([num_csv_col-1], tf.int64)}
    parsed_feature = tf.parse_single_example(serialized_example, features=feature_set)
    parsed_label = parsed_feature['label']
    parsed_data = parsed_feature['data']
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3*batch_size
    feature_batch, label_batch = tf.train.batch([parsed_data, parsed_label], \
            batch_size=batch_size, capacity=capacity)
    return feature_batch, label_batch      
    
time_style = '%Y-%m-%d %H:%M:%S'
def print_time():
    now = datetime.datetime.now()
    time_str = now.strftime(time_style)
    print(time_str)

